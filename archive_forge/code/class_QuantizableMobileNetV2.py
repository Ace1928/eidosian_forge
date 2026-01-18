from functools import partial
from typing import Any, Optional, Union
from torch import nn, Tensor
from torch.ao.quantization import DeQuantStub, QuantStub
from torchvision.models.mobilenetv2 import InvertedResidual, MobileNet_V2_Weights, MobileNetV2
from ...ops.misc import Conv2dNormActivation
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from .utils import _fuse_modules, _replace_relu, quantize_model
class QuantizableMobileNetV2(MobileNetV2):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        MobileNet V2 main class

        Args:
           Inherits args from floating point MobileNetV2
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
                _fuse_modules(m, ['0', '1', '2'], is_qat, inplace=True)
            if type(m) is QuantizableInvertedResidual:
                m.fuse_model(is_qat)