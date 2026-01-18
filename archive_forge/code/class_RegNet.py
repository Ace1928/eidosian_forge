import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from ..ops.misc import Conv2dNormActivation, SqueezeExcitation
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
class RegNet(nn.Module):

    def __init__(self, block_params: BlockParams, num_classes: int=1000, stem_width: int=32, stem_type: Optional[Callable[..., nn.Module]]=None, block_type: Optional[Callable[..., nn.Module]]=None, norm_layer: Optional[Callable[..., nn.Module]]=None, activation: Optional[Callable[..., nn.Module]]=None) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU
        self.stem = stem_type(3, stem_width, norm_layer, activation)
        current_width = stem_width
        blocks = []
        for i, (width_out, stride, depth, group_width, bottleneck_multiplier) in enumerate(block_params._get_expanded_params()):
            blocks.append((f'block{i + 1}', AnyStage(current_width, width_out, stride, depth, block_type, norm_layer, activation, group_width, bottleneck_multiplier, block_params.se_ratio, stage_index=i + 1)))
            current_width = width_out
        self.trunk_output = nn.Sequential(OrderedDict(blocks))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=current_width, out_features=num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.trunk_output(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x