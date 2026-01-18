from functools import partial
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import shufflenetv2
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..shufflenetv2 import (
from .utils import _fuse_modules, _replace_relu, quantize_model
class QuantizableShuffleNetV2(shufflenetv2.ShuffleNetV2):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, inverted_residual=QuantizableInvertedResidual, **kwargs)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self, is_qat: Optional[bool]=None) -> None:
        """Fuse conv/bn/relu modules in shufflenetv2 model

        Fuse conv+bn+relu/ conv+relu/conv+bn modules to prepare for quantization.
        Model is modified in place.

        .. note::
            Note that this operation does not change numerics
            and the model after modification is in floating point
        """
        for name, m in self._modules.items():
            if name in ['conv1', 'conv5'] and m is not None:
                _fuse_modules(m, [['0', '1', '2']], is_qat, inplace=True)
        for m in self.modules():
            if type(m) is QuantizableInvertedResidual:
                if len(m.branch1._modules.items()) > 0:
                    _fuse_modules(m.branch1, [['0', '1'], ['2', '3', '4']], is_qat, inplace=True)
                _fuse_modules(m.branch2, [['0', '1', '2'], ['3', '4'], ['5', '6', '7']], is_qat, inplace=True)