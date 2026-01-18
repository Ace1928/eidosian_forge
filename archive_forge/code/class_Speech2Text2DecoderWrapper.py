import copy
import math
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging, replace_return_docstrings
from .configuration_speech_to_text_2 import Speech2Text2Config
@add_start_docstrings('The Speech2Text2 Model with a language modeling head. Can be used for summarization.', SPEECH_TO_TEXT_2_START_DOCSTRING)
class Speech2Text2DecoderWrapper(Speech2Text2PreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        super().__init__(config)
        self.decoder = Speech2Text2Decoder(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)