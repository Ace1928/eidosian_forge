from collections import defaultdict
from collections.abc import Iterable
import numpy as np
import torch
import hypothesis
from functools import reduce
from hypothesis import assume
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as stnp
from hypothesis.strategies import SearchStrategy
from torch.testing._internal.common_quantized import _calculate_dynamic_qparams, _calculate_dynamic_per_channel_qparams
def _get_valid_min_max(qparams):
    scale, zero_point, quantized_type = qparams
    adjustment = 1 + torch.finfo(torch.float).eps
    _long_type_info = torch.iinfo(torch.long)
    long_min, long_max = (_long_type_info.min / adjustment, _long_type_info.max / adjustment)
    min_value = max((long_min - zero_point) * scale, long_min / scale + zero_point)
    max_value = min((long_max - zero_point) * scale, long_max / scale + zero_point)
    return (np.float32(min_value), np.float32(max_value))