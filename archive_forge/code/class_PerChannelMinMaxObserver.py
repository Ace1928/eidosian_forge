import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
class PerChannelMinMaxObserver(UniformQuantizationObserverBase):
    """Observer module for computing the quantization parameters based on the
    running per channel min and max values.

    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the running minimum and maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        ch_axis: Channel axis
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The quantization parameters are computed the same way as in
    :class:`~torch.ao.quantization.observer.MinMaxObserver`, with the difference
    that the running min/max values are stored per channel.
    Scales and zero points are thus computed per channel as well.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self, ch_axis=0, dtype=torch.quint8, qscheme=torch.per_channel_affine, reduce_range=False, quant_min=None, quant_max=None, factory_kwargs=None, eps=torch.finfo(torch.float32).eps, is_dynamic=False, **kwargs) -> None:
        if not is_per_channel(qscheme):
            raise NotImplementedError("PerChannelMinMaxObserver's qscheme only support                     torch.per_channel_symmetric, torch.per_channel_affine and torch.per_channel_affine_float_qparams.")
        if is_dynamic:
            raise NotImplementedError("PerChannelMinMaxObserver doesn't support dynamic quantization")
        super().__init__(dtype=dtype, qscheme=qscheme, reduce_range=reduce_range, quant_min=quant_min, quant_max=quant_max, factory_kwargs=factory_kwargs, eps=eps, is_dynamic=is_dynamic, **kwargs)
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.ch_axis = ch_axis
        self.register_buffer('min_val', torch.tensor([], **factory_kwargs))
        self.register_buffer('max_val', torch.tensor([], **factory_kwargs))
        if self.qscheme == torch.per_channel_symmetric and self.reduce_range and (self.dtype == torch.quint8):
            raise NotImplementedError('Cannot reduce range for symmetric quantization for quint8')

    def forward(self, x_orig):
        return self._forward(x_orig)

    def _forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x.size()
        new_axis_list = [i for i in range(len(x_dim))]
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        y = y.to(self.min_val.dtype)
        y = torch.flatten(y, start_dim=1)
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = torch.aminmax(y, dim=1)
        else:
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
            min_val = torch.min(min_val_cur, min_val)
            max_val = torch.max(max_val_cur, max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        return self._calculate_qparams(self.min_val, self.max_val)

    def extra_repr(self):
        return f'min_val={self.min_val}, max_val={self.max_val}'

    def _load_from_state_dict(self, state_dict: Dict[str, Any], prefix: str, local_metadata: Dict[str, torch.Tensor], strict: bool, missing_keys: List[str], unexpected_keys: List[str], error_msgs: List[str]):
        version = local_metadata.get('version', None)
        if version is None or version < 3:
            local_state = ['min_vals', 'max_vals']
            expected_min_name = 'min_vals'
            expected_max_name = 'max_vals'
        else:
            local_state = ['min_val', 'max_val']
            expected_min_name = 'min_val'
            expected_max_name = 'max_val'
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                if name == expected_min_name:
                    self.min_val.resize_(val.shape)
                elif name == expected_max_name:
                    self.max_val.resize_(val.shape)
                else:
                    warnings.warn(f'Observer load_from_state_dict got unexpected name {name}')
                if torch.jit.is_scripting():
                    if name == expected_min_name:
                        self.min_val.copy_(val)
                    elif name == expected_max_name:
                        self.max_val.copy_(val)
                    else:
                        warnings.warn(f'Observer load_from_state_dict got unexpected name {name}')
            elif strict:
                missing_keys.append(key)
        if not torch.jit.is_scripting():
            super()._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)

    def _load_from_state_dict_script(self, state_dict: Dict[str, Any], prefix: str, local_metadata: Dict[str, torch.Tensor], strict: bool, missing_keys: List[str], unexpected_keys: List[str], error_msgs: List[str]):
        self._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        self.min_val = torch.rand(0)
        self.max_val = torch.rand(0)