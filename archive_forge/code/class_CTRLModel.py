from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_ctrl import CTRLConfig
@add_start_docstrings('The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.', CTRL_START_DOCSTRING)
class CTRLModel(CTRLPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.d_model_size = config.n_embd
        self.num_layers = config.n_layer
        self.pos_encoding = positional_encoding(config.n_positions, self.d_model_size, torch.float)
        self.w = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([EncoderLayer(config.n_embd, config.n_head, config.dff, config.resid_pdrop) for _ in range(config.n_layer)])
        self.layernorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.post_init()

    def get_input_embeddings(self):
        return self.w

    def set_input_embeddings(self, new_embeddings):
        self.w = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].multi_head_attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPast]:
        """
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, CTRLModel
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
        >>> model = CTRLModel.from_pretrained("Salesforce/ctrl")

        >>> # CTRL was trained with control codes as the first token
        >>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")
        >>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 5, 1280]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError('batch_size has to be defined and > 0')
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
            token_type_embeds = self.w(token_type_ids)
            token_type_embeds *= np.sqrt(self.d_model_size)
        else:
            token_type_embeds = 0
        if inputs_embeds is None:
            inputs_embeds = self.w(input_ids)
        seq_len = input_shape[-1]
        mask = torch.triu(torch.ones(seq_len + past_length, seq_len + past_length), 1).to(device)
        inputs_embeds *= np.sqrt(self.d_model_size)
        self.pos_encoding = self.pos_encoding.to(device)
        pos_embeds = self.pos_encoding[position_ids, :]
        hidden_states = inputs_embeds + pos_embeds + token_type_embeds
        hidden_states = self.dropout(hidden_states)
        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, (h, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = h(hidden_states, mask, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[i], use_cache=use_cache, output_attentions=output_attentions)
            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)
            if output_attentions:
                all_attentions += (outputs[2],)
        hidden_states = self.layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None))
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_attentions)