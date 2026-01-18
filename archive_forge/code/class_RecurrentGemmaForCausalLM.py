import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import BaseModelOutputWithNoAttention, CausalLMOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_recurrent_gemma import RecurrentGemmaConfig
class RecurrentGemmaForCausalLM(RecurrentGemmaPreTrainedModel):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config):
        super().__init__(config)
        self.model = RecurrentGemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(RECURRENTGEMMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, cache_position: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, use_cache: Optional[bool]=None, **kwargs) -> Union[Tuple, CausalLMOutput]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RecurrentGemmaForCausalLM

        >>> model = RecurrentGemmaForCausalLM.from_pretrained("google/recurrentgemma-2b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/recurrentgemma-2b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True
        outputs = self.model(input_ids=input_ids, cache_position=cache_position, attention_mask=attention_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        cap = self.config.logits_soft_cap
        logits = nn.functional.tanh(logits / cap) * cap
        logits = logits.float()
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, inputs_embeds=None, cache_position=None, use_cache=None, **kwargs):
        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        attention_mask = attention_mask[:, -self.config.attention_window_size:]
        past_length = cache_position[0]
        if past_length > 0:
            position_ids = position_ids[:, past_length:]
        if inputs_embeds is not None:
            model_inputs = {'inputs_embeds': inputs_embeds[:, past_length:]}
        else:
            model_inputs = {'input_ids': input_ids[:, past_length:].contiguous()}
        if cache_position is not None:
            cache_position = cache_position[-position_ids.shape[1]:]
        model_inputs.update({'position_ids': position_ids, 'attention_mask': attention_mask, 'cache_position': cache_position, 'use_cache': use_cache})
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        for layer in self.layers:
            if hasattr(layer.temporal_block, 'key_states'):
                k_state = layer.temporal_block.key_states
                v_state = layer.temporal_block.value_states
                k_state = k_state.index_select(0, beam_idx.to(k_state.device))
                v_state = v_state.index_select(0, beam_idx.to(v_state.device))
        return None