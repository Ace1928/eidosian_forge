from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
@dataclass
class DTypeWithConstraints:
    """
    Config for specifying additional constraints for a given dtype, such as quantization
    value ranges, scale value ranges, and fixed quantization params, to be used in
    :class:`~torch.ao.quantization.backend_config.DTypeConfig`.

    The constraints currently supported are:

    * `quant_min_lower_bound` and `quant_max_upper_bound`: Lower and upper
      bounds for the minimum and maximum quantized values respectively. If
      the QConfig’s `quant_min` and `quant_max` fall outside this range,
      then the QConfig will be ignored.

    * `scale_min_lower_bound` and `scale_max_upper_bound`: Lower and upper
      bounds for the minimum and maximum scale values respectively. If the
      QConfig’s minimum scale value (currently exposed as `eps`) falls below
      the lower bound, then the QConfig will be ignored. Note that the upper
      bound is currently not enforced.

    * `scale_exact_match` and `zero_point_exact_match`: Exact match requirements
      for scale and zero point, to be used for operators with fixed quantization
      parameters such as sigmoid and tanh. If the observer specified in the QConfig
      is neither `FixedQParamsObserver` nor `FixedQParamsFakeQuantize`, or if
      the quantization parameters don't match, then the QConfig will be ignored.
    """
    dtype: Optional[torch.dtype] = None
    quant_min_lower_bound: Union[int, float, None] = None
    quant_max_upper_bound: Union[int, float, None] = None
    scale_min_lower_bound: Union[int, float, None] = None
    scale_max_upper_bound: Union[int, float, None] = None
    scale_exact_match: Optional[float] = None
    zero_point_exact_match: Optional[int] = None