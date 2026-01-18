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
class QuantizableInceptionD(inception_module.InceptionD):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, conv_block=QuantizableBasicConv2d, **kwargs)
        self.myop = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)