import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import LayerNorm
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_xlm_prophetnet import XLMProphetNetConfig
class XLMProphetNetDecoderWrapper(XLMProphetNetPreTrainedModel):
    """
    This is a wrapper class, so that [`XLMProphetNetForCausalLM`] can correctly be loaded from pretrained XLMProphetNet
    classes.
    """

    def __init__(self, config: XLMProphetNetConfig):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.decoder = XLMProphetNetDecoder(config, word_embeddings=self.word_embeddings)
        self.post_init()

    def _tie_weights(self):
        self._tie_or_clone_weights(self.word_embeddings, self.decoder.get_input_embeddings())

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)