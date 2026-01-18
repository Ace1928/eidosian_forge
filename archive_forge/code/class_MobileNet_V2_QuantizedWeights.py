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
class MobileNet_V2_QuantizedWeights(WeightsEnum):
    IMAGENET1K_QNNPACK_V1 = Weights(url='https://download.pytorch.org/models/quantized/mobilenet_v2_qnnpack_37f702c5.pth', transforms=partial(ImageClassification, crop_size=224), meta={'num_params': 3504872, 'min_size': (1, 1), 'categories': _IMAGENET_CATEGORIES, 'backend': 'qnnpack', 'recipe': 'https://github.com/pytorch/vision/tree/main/references/classification#qat-mobilenetv2', 'unquantized': MobileNet_V2_Weights.IMAGENET1K_V1, '_metrics': {'ImageNet-1K': {'acc@1': 71.658, 'acc@5': 90.15}}, '_ops': 0.301, '_file_size': 3.423, '_docs': '\n                These weights were produced by doing Quantization Aware Training (eager mode) on top of the unquantized\n                weights listed below.\n            '})
    DEFAULT = IMAGENET1K_QNNPACK_V1