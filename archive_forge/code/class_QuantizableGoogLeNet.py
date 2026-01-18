import warnings
from functools import partial
from typing import Any, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..googlenet import BasicConv2d, GoogLeNet, GoogLeNet_Weights, GoogLeNetOutputs, Inception, InceptionAux
from .utils import _fuse_modules, _replace_relu, quantize_model
class QuantizableGoogLeNet(GoogLeNet):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, blocks=[QuantizableBasicConv2d, QuantizableInception, QuantizableInceptionAux], **kwargs)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: Tensor) -> GoogLeNetOutputs:
        x = self._transform_input(x)
        x = self.quant(x)
        x, aux1, aux2 = self._forward(x)
        x = self.dequant(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn('Scripted QuantizableGoogleNet always returns GoogleNetOutputs Tuple')
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)

    def fuse_model(self, is_qat: Optional[bool]=None) -> None:
        """Fuse conv/bn/relu modules in googlenet model

        Fuse conv+bn+relu/ conv+relu/conv+bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """
        for m in self.modules():
            if type(m) is QuantizableBasicConv2d:
                m.fuse_model(is_qat)