import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
@add_start_docstrings('The bare MEGA Model transformer outputting raw hidden-states without any specific head on top.', MEGA_START_DOCSTRING)
class MegaModel(MegaPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added after self-attention, following the architecture described in *Mega: Moving Average
    Equipped Gated Attention*_ by Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig,
    Jonathan May, and Luke Zettlemoyer

    To behave as a decoder the model needs to be initialized with the `is_decoder` argument of the configuration set to
    `True` and `bidirectional` set to `False`. To be used in a Seq2Seq model, the model needs to initialized with both
    `is_decoder=True` and `bidirectional=False` argument as well as `add_cross_attention` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Mega: Moving Average Equipped Gated Attention*: https://arxiv.org/abs/2209.10655

    """

    def __init__(self, config: MegaConfig, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embedding_layer = MegaEmbeddings(config)
        self.layers = nn.ModuleList([MegaBlock(config) for _ in range(config.num_hidden_layers)])
        self.pooler = MegaPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding_layer.word_embeddings

    def set_input_embeddings(self, value):
        self.embedding_layer.word_embeddings = value

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if self.config.use_chunking:
            input_shape = torch.tensor([input_shape[0], self.config.chunk_size])
        batch_size, sequence_length = input_shape
        if self.config.use_chunking and sequence_length > self.config.chunk_size:
            if sequence_length % self.config.chunk_size != 0:
                raise ValueError(f'config.use_chunking is activated; input sequence length must be shorter than or a multiple of config.chunk_size\nreceived sequence length of {sequence_length} with chunk size {self.config.chunk_size}')
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            temp_mask_for_extension = torch.ones((1, sequence_length), dtype=torch.long, device=device)
            causal_mask = self.create_extended_attention_mask_for_decoder(input_shape, temp_mask_for_extension)
            causal_mask = causal_mask.squeeze(0)
        else:
            use_cache = False
            causal_mask = None
        if past_key_values is not None and len(past_key_values) != self.config.num_hidden_layers:
            raise ValueError(f'Received past key/value cache with size mismatch; expected {self.config.num_hidden_layers}, received {len(past_key_values)}')
        embedding_output = self.embedding_layer(input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        hidden_states = embedding_output.transpose(0, 1)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        all_hidden_states = (embedding_output,) if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None
        for i, mega_layer in enumerate(self.layers):
            current_decoder_cache = past_key_values[i] if past_key_values is not None else None
            mega_outputs = mega_layer(hidden_states=hidden_states, attention_mask=attention_mask, causal_mask=causal_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_value=current_decoder_cache, output_attentions=output_attentions, use_cache=use_cache)
            hidden_states = mega_outputs[0]
            if output_hidden_states:
                all_hidden_states += (hidden_states.transpose(0, 1),)
            if output_attentions:
                self_attn_weights = mega_outputs[1]
                all_self_attentions += (self_attn_weights,)
                if self.config.add_cross_attention:
                    cross_attn_weights = mega_outputs[2]
                    all_cross_attentions += (cross_attn_weights,)
            if use_cache:
                updated_cache = mega_outputs[-1]
                next_decoder_cache += (updated_cache,)
        hidden_states = hidden_states.transpose(0, 1)
        pooled_output = self.pooler(hidden_states) if self.pooler is not None else None
        if not return_dict:
            return (hidden_states, pooled_output) + (all_hidden_states, next_decoder_cache, all_self_attentions, all_cross_attentions)
        return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=hidden_states, pooler_output=pooled_output, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)