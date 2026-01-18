import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mgp_str import MgpstrConfig
class MgpstrPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MgpstrConfig
    base_model_prefix = 'mgp_str'

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, MgpstrEmbeddings):
            nn.init.trunc_normal_(module.pos_embed, mean=0.0, std=self.config.initializer_range)
            nn.init.trunc_normal_(module.cls_token, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)