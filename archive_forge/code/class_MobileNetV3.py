from functools import partial
from typing import Any, Callable, List, Optional, Sequence
import torch
from torch import nn, Tensor
from ..ops.misc import Conv2dNormActivation, SqueezeExcitation as SElayer
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
class MobileNetV3(nn.Module):

    def __init__(self, inverted_residual_setting: List[InvertedResidualConfig], last_channel: int, num_classes: int=1000, block: Optional[Callable[..., nn.Module]]=None, norm_layer: Optional[Callable[..., nn.Module]]=None, dropout: float=0.2, **kwargs: Any) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            dropout (float): The droupout probability
        """
        super().__init__()
        _log_api_usage_once(self)
        if not inverted_residual_setting:
            raise ValueError('The inverted_residual_setting should not be empty')
        elif not (isinstance(inverted_residual_setting, Sequence) and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError('The inverted_residual_setting should be List[InvertedResidualConfig]')
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        layers: List[nn.Module] = []
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(Conv2dNormActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.Hardswish))
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(Conv2dNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_channels, last_channel), nn.Hardswish(inplace=True), nn.Dropout(p=dropout, inplace=True), nn.Linear(last_channel, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)