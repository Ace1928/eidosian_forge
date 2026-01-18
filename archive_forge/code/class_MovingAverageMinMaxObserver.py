import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
class MovingAverageMinMaxObserver(MinMaxObserver):
    """Observer module for computing the quantization parameters based on the
    moving average of the min and max values.

    This observer computes the quantization parameters based on the moving
    averages of minimums and maximums of the incoming tensors. The module
    records the average minimum and maximum of incoming tensors, and uses this
    statistic to compute the quantization parameters.

    Args:
        averaging_constant: Averaging constant for min/max.
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The moving average min/max is computed as follows

    .. math::

        \\begin{array}{ll}
                x_\\text{min} = \\begin{cases}
                    \\min(X) & \\text{if~}x_\\text{min} = \\text{None} \\\\
                    (1 - c) x_\\text{min} + c \\min(X) & \\text{otherwise}
                \\end{cases}\\\\
                x_\\text{max} = \\begin{cases}
                    \\max(X) & \\text{if~}x_\\text{max} = \\text{None} \\\\
                    (1 - c) x_\\text{max} + c \\max(X) & \\text{otherwise}
                \\end{cases}\\\\
        \\end{array}

    where :math:`x_\\text{min/max}` is the running average min/max, :math:`X` is
    is the incoming tensor, and :math:`c` is the ``averaging_constant``.

    The scale and zero point are then computed as in
    :class:`~torch.ao.quantization.observer.MinMaxObserver`.

    .. note:: Only works with ``torch.per_tensor_affine`` quantization scheme.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """

    def __init__(self, averaging_constant=0.01, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False, quant_min=None, quant_max=None, eps=torch.finfo(torch.float32).eps, is_dynamic=False, **kwargs) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(f"MovingAverageMinMaxObserver's qscheme only support                 torch.per_tensor_symmetric and torch.per_tensor_affine.                 but got: {qscheme}")
        self.averaging_constant = averaging_constant
        if is_dynamic and self.averaging_constant != 1:
            raise NotImplementedError(f"MovingAverageMinMaxObserver doesn't support dynamic quantization for averaging constant of {self.averaging_constant}")
        super().__init__(dtype=dtype, qscheme=qscheme, reduce_range=reduce_range, quant_min=quant_min, quant_max=quant_max, eps=eps, is_dynamic=is_dynamic, **kwargs)

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        if min_val == float('inf') and max_val == float('-inf'):
            min_val, max_val = torch.aminmax(x)
        else:
            min_val_cur, max_val_cur = torch.aminmax(x)
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig