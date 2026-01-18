import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
class UniformQuantizationObserverBase(ObserverBase):
    """Common base for all observers using uniform quantization to calculate
    scale and zero_point.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used.
        reduce_range: Reduces the range of the quantized data type by 1 bit.
                      This is sometimes required to avoid instruction overflow.
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    .. warning::

        :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.
               or `torch.int8` or `torch.uint8`

    .. warning::

        :attr:`qscheme` can only take one of the following options:

        - ``torch.per_tensor_affine``
        - ``torch.per_tensor_symmetric``
        - ``torch.per_channel_affine``
        - ``torch.per_channel_symmetric``
    """
    _version = 3
    eps: torch.Tensor

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False, quant_min=None, quant_max=None, factory_kwargs=None, eps=torch.finfo(torch.float32).eps, is_dynamic=False, **kwargs) -> None:
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        super().__init__(dtype=dtype, is_dynamic=is_dynamic, **kwargs)
        self.qscheme = qscheme
        if reduce_range:
            warnings.warn('Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.')
        self.reduce_range = reduce_range
        self.register_buffer('eps', torch.tensor([eps], **factory_kwargs))
        assert self.qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric, torch.per_channel_affine, torch.per_channel_symmetric, torch.per_channel_affine_float_qparams), 'Default Observer only works for per_tensor_affine,                 per_tensor_symmetric, per_channel_affine,                 per_channel_symmetric and per_channel_float_qparams quantization scheme'
        _ALLOWED_DTYPES = (torch.qint8, torch.quint8, torch.quint4x2, torch.qint32, torch.int8, torch.uint8, torch.int16, torch.int32)
        assert self.dtype in _ALLOWED_DTYPES, f'Default Observer only works for {_ALLOWED_DTYPES} data type'
        self.has_customized_qrange = quant_min is not None and quant_max is not None
        if self.has_customized_qrange:
            validate_qmin_qmax(quant_min, quant_max)
        self.quant_min, self.quant_max = calculate_qmin_qmax(quant_min, quant_max, self.has_customized_qrange, self.dtype, self.reduce_range)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version == 1:
            eps = torch.tensor([torch.finfo(torch.float32).eps])
            state_dict[prefix + 'eps'] = eps
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @torch.jit.export
    def _validate_qmin_qmax(self, quant_min: int, quant_max: int) -> None:
        """Validates that the user-specified quantization range is properly initialized
        and within the given bound supported by the observer dtype.

        To accommodate lower-bit quantization with respect to the existing torch.qint8 and
        torch.quint8 datatypes, the user can choose to use dynamic quantization range by passing
        in a tuple of initial qmin and qmax values. One use case is these customized qmin and qmax
        values are used to calculate static estimates of the scale and zero point for aggressive lower-bit
        fake quantization. These estimates are compared against parameters learned through backpropagation.
        The related literatures for scale and zero point via backpropagation are as follows:

        Learned Step Size Quantization: https://openreview.net/pdf?id=rkgO66VKDS
        Trained Quantization Thresholds: https://arxiv.org/pdf/1903.08066.pdf
        """
        assert quant_min <= 0 <= quant_max, 'Used-specified quantization range must include 0.'
        assert quant_min < quant_max, 'qmin must be strictly less than qmax for user-specified quantization range.'

    @torch.jit.export
    def _calculate_qparams(self, min_val: torch.Tensor, max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the quantization parameters, given min and max
        value tensors. Works for both per tensor and per channel cases

        Args:
            min_val: Minimum values per channel
            max_val: Maximum values per channel

        Returns:
            scales: Scales tensor of shape (#channels,)
            zero_points: Zero points tensor of shape (#channels,)
        """
        if not check_min_max_valid(min_val, max_val):
            return (torch.tensor([1.0], device=min_val.device.type), torch.tensor([0], device=min_val.device.type))
        quant_min, quant_max = (self.quant_min, self.quant_max)
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)
        if self.qscheme == torch.per_tensor_symmetric or self.qscheme == torch.per_channel_symmetric:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
            if self.dtype in [torch.quint8, torch.uint8]:
                if self.has_customized_qrange:
                    zero_point = zero_point.new_full(zero_point.size(), (quant_min + quant_max) // 2)
                else:
                    zero_point = zero_point.new_full(zero_point.size(), 128)
        elif self.qscheme == torch.per_channel_affine_float_qparams:
            scale = (max_val - min_val) / float(quant_max - quant_min)
            scale = torch.where(scale > self.eps, scale, torch.ones_like(scale))
            zero_point = -1 * min_val / scale
        else:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
        if len(scale.shape) == 0:
            scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
        if len(zero_point.shape) == 0:
            zero_point = torch.tensor([int(zero_point)], dtype=zero_point.dtype, device=device)
            if self.qscheme == torch.per_channel_affine_float_qparams:
                zero_point = torch.tensor([float(zero_point)], dtype=zero_point.dtype, device=device)
        return (scale, zero_point)

    @torch.jit.export
    def reset_min_max_vals(self):
        raise NotImplementedError('Cannot reset min/max values in the given observer.')