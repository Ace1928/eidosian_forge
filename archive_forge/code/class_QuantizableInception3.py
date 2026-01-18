import warnings
from functools import partial
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import inception as inception_module
from torchvision.models.inception import Inception_V3_Weights, InceptionOutputs
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from .utils import _fuse_modules, _replace_relu, quantize_model
class QuantizableInception3(inception_module.Inception3):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, inception_blocks=[QuantizableBasicConv2d, QuantizableInceptionA, QuantizableInceptionB, QuantizableInceptionC, QuantizableInceptionD, QuantizableInceptionE, QuantizableInceptionAux], **kwargs)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: Tensor) -> InceptionOutputs:
        x = self._transform_input(x)
        x = self.quant(x)
        x, aux = self._forward(x)
        x = self.dequant(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn('Scripted QuantizableInception3 always returns QuantizableInception3 Tuple')
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)

    def fuse_model(self, is_qat: Optional[bool]=None) -> None:
        """Fuse conv/bn/relu modules in inception model

        Fuse conv+bn+relu/ conv+relu/conv+bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """
        for m in self.modules():
            if type(m) is QuantizableBasicConv2d:
                m.fuse_model(is_qat)