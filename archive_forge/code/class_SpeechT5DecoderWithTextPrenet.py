import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig
class SpeechT5DecoderWithTextPrenet(SpeechT5PreTrainedModel):
    """
    Wrapper around SpeechT5Decoder that applies SpeechT5TextDecoderPrenet to convert input tokens to hidden features.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        self.prenet = SpeechT5TextDecoderPrenet(config)
        self.wrapped_decoder = SpeechT5Decoder(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.prenet.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.prenet.set_input_embeddings(value)

    def forward(self, input_values: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        decoder_hidden_states, attention_mask = self.prenet(input_values, attention_mask, past_key_values)
        outputs = self.wrapped_decoder(hidden_states=decoder_hidden_states, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, head_mask=head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        return outputs