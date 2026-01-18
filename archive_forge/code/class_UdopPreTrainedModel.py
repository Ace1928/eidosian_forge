import collections
import logging
import math
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import UdopConfig
from transformers.modeling_outputs import (
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ..deprecated._archive_maps import UDOP_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
class UdopPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models. Based on `T5PreTrainedModel`.
    """
    config_class = UdopConfig
    base_model_prefix = 'transformer'
    supports_gradient_checkpointing = True
    _keep_in_fp32_modules = ['wo']

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, UdopLayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data.to(torch.float32), mean=0.0, std=factor).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, RelativePositionBiasBase):
            factor = self.config.initializer_factor
            d_model = self.config.d_model
            module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * d_model ** (-0.5))
        elif isinstance(module, UdopModel):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, UdopForConditionalGeneration):
            if hasattr(module, 'lm_head') and (not self.config.tie_word_embeddings):
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, UdopDenseActDense):
            module.wi.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module.wi, 'bias') and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * self.config.d_ff ** (-0.5))
            if hasattr(module.wo, 'bias') and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, UdopDenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module.wi_0, 'bias') and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module.wi_1, 'bias') and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * self.config.d_ff ** (-0.5))
            if hasattr(module.wo, 'bias') and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, UdopAttention):
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * (d_model * key_value_proj_dim) ** (-0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * d_model ** (-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * d_model ** (-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * (n_heads * key_value_proj_dim) ** (-0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * d_model ** (-0.5))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        assert decoder_start_token_id is not None, 'self.model.config.decoder_start_token_id has to be defined. In Udop it is usually set to the pad_token_id. See Udop docs for more information'
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id
        assert pad_token_id is not None, 'self.model.config.pad_token_id has to be defined.'
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        assert torch.all(shifted_input_ids >= 0).item(), 'Verify that `shifted_input_ids` has only positive values'
        return shifted_input_ids