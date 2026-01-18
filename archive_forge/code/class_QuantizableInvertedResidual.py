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
class QuantizableInvertedResidual(InvertedResidual):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

    def fuse_model(self, is_qat: Optional[bool]=None) -> None:
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) is nn.Conv2d:
                _fuse_modules(self.conv, [str(idx), str(idx + 1)], is_qat, inplace=True)