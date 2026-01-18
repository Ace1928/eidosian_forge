import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...activations import ACT2CLS
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_patchtst import PatchTSTConfig
class PatchTSTPreTrainedModel(PreTrainedModel):
    config_class = PatchTSTConfig
    base_model_prefix = 'model'
    main_input_name = 'past_values'
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """
        Initialize weights
        """
        if isinstance(module, PatchTSTPositionalEncoding):
            if self.config.use_cls_token:
                nn.init.normal_(module.cls_token, std=0.02)
            if self.config.positional_encoding_type == 'random':
                nn.init.normal_(module.position_enc, mean=0.0, std=0.1)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, PatchTSTBatchNorm):
            module.batchnorm.bias.data.zero_()
            module.batchnorm.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, PatchTSTEncoder):
            module.gradient_checkpointing = value