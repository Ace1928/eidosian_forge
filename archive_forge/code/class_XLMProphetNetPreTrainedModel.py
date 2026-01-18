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
class XLMProphetNetPreTrainedModel(PreTrainedModel):
    config_class = XLMProphetNetConfig
    base_model_prefix = 'prophetnet'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        assert decoder_start_token_id is not None, 'self.model.config.decoder_start_token_id has to be defined. In XLMProphetNet it is usually set to the pad_token_id. See XLMProphetNet docs for more information'
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id
        assert pad_token_id is not None, 'self.model.config.pad_token_id has to be defined.'
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        assert torch.all(shifted_input_ids >= 0).item(), 'Verify that `shifted_input_ids` has only positive values'
        return shifted_input_ids