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
class QuantizableSqueezeExcitation(SqueezeExcitation):
    _version = 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs['scale_activation'] = nn.Hardsigmoid
        super().__init__(*args, **kwargs)
        self.skip_mul = nn.quantized.FloatFunctional()

    def forward(self, input: Tensor) -> Tensor:
        return self.skip_mul.mul(self._scale(input), input)

    def fuse_model(self, is_qat: Optional[bool]=None) -> None:
        _fuse_modules(self, ['fc1', 'activation'], is_qat, inplace=True)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if hasattr(self, 'qconfig') and (version is None or version < 2):
            default_state_dict = {'scale_activation.activation_post_process.scale': torch.tensor([1.0]), 'scale_activation.activation_post_process.activation_post_process.scale': torch.tensor([1.0]), 'scale_activation.activation_post_process.zero_point': torch.tensor([0], dtype=torch.int32), 'scale_activation.activation_post_process.activation_post_process.zero_point': torch.tensor([0], dtype=torch.int32), 'scale_activation.activation_post_process.fake_quant_enabled': torch.tensor([1]), 'scale_activation.activation_post_process.observer_enabled': torch.tensor([1])}
            for k, v in default_state_dict.items():
                full_key = prefix + k
                if full_key not in state_dict:
                    state_dict[full_key] = v
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)