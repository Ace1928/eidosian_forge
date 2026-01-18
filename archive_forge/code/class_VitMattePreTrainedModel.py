from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_vitmatte import VitMatteConfig
class VitMattePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = VitMatteConfig
    main_input_name = 'pixel_values'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()