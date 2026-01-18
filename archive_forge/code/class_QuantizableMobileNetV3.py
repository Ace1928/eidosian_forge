from functools import partial
from typing import Any, List, Optional, Union
import torch
from torch import nn, Tensor
from torch.ao.quantization import DeQuantStub, QuantStub
from ...ops.misc import Conv2dNormActivation, SqueezeExcitation
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..mobilenetv3 import (
from .utils import _fuse_modules, _replace_relu
class QuantizableMobileNetV3(MobileNetV3):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        MobileNet V3 main class

        Args:
           Inherits args from floating point MobileNetV3
        """
        super().__init__(*args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self, is_qat: Optional[bool]=None) -> None:
        for m in self.modules():
            if type(m) is Conv2dNormActivation:
                modules_to_fuse = ['0', '1']
                if len(m) == 3 and type(m[2]) is nn.ReLU:
                    modules_to_fuse.append('2')
                _fuse_modules(m, modules_to_fuse, is_qat, inplace=True)
            elif type(m) is QuantizableSqueezeExcitation:
                m.fuse_model(is_qat)