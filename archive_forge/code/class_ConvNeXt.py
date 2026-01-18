from functools import partial
from typing import Any, Callable, List, Optional, Sequence
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from ..ops.misc import Conv2dNormActivation, Permute
from ..ops.stochastic_depth import StochasticDepth
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class ConvNeXt(nn.Module):

    def __init__(self, block_setting: List[CNBlockConfig], stochastic_depth_prob: float=0.0, layer_scale: float=1e-06, num_classes: int=1000, block: Optional[Callable[..., nn.Module]]=None, norm_layer: Optional[Callable[..., nn.Module]]=None, **kwargs: Any) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if not block_setting:
            raise ValueError('The block_setting should not be empty')
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError('The block_setting should be List[CNBlockConfig]')
        if block is None:
            block = CNBlock
        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-06)
        layers: List[nn.Module] = []
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(Conv2dNormActivation(3, firstconv_output_channels, kernel_size=4, stride=4, padding=0, norm_layer=norm_layer, activation_layer=None, bias=True))
        total_stage_blocks = sum((cnf.num_layers for cnf in block_setting))
        stage_block_id = 0
        for cnf in block_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                layers.append(nn.Sequential(norm_layer(cnf.input_channels), nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2)))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        lastblock = block_setting[-1]
        lastconv_output_channels = lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        self.classifier = nn.Sequential(norm_layer(lastconv_output_channels), nn.Flatten(1), nn.Linear(lastconv_output_channels, num_classes))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)