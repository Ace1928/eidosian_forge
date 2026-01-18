import collections.abc
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_vitdet import VitDetConfig
class VitDetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = VitDetConfig
    base_model_prefix = 'vitdet'
    main_input_name = 'pixel_values'
    supports_gradient_checkpointing = True
    _no_split_modules = []

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, VitDetEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(module.position_embeddings.data.to(torch.float32), mean=0.0, std=self.config.initializer_range).to(module.position_embeddings.dtype)
        elif isinstance(module, VitDetAttention) and self.config.use_relative_position_embeddings:
            module.rel_pos_h.data = nn.init.trunc_normal_(module.rel_pos_h.data.to(torch.float32), mean=0.0, std=self.config.initializer_range)
            module.rel_pos_w.data = nn.init.trunc_normal_(module.rel_pos_w.data.to(torch.float32), mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, VitDetResBottleneckBlock):
            for layer in [module.conv1, module.conv2, module.conv3]:
                caffe2_msra_fill(layer)
            for layer in [module.norm1, module.norm2]:
                layer.weight.data.fill_(1.0)
                layer.bias.data.zero_()
            module.norm3.weight.data.zero_()
            module.norm3.bias.data.zero_()