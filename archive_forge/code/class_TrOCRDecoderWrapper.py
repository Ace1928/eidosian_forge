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
from .configuration_trocr import TrOCRConfig
@add_start_docstrings('The TrOCR Model with a language modeling head. Can be used for summarization.', TROCR_START_DOCSTRING)
class TrOCRDecoderWrapper(TrOCRPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        super().__init__(config)
        self.decoder = TrOCRDecoder(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)