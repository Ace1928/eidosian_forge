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
def _floats_wrapper(*args, **kwargs):
    if 'width' in kwargs and hypothesis.version.__version_info__ < (3, 67, 0):
        no_nan_and_inf = ('allow_nan' in kwargs and (not kwargs['allow_nan']) or 'allow_nan' not in kwargs) and ('allow_infinity' in kwargs and (not kwargs['allow_infinity']) or 'allow_infinity' not in kwargs)
        min_and_max_not_specified = len(args) == 0 and 'min_value' not in kwargs and ('max_value' not in kwargs)
        if no_nan_and_inf and min_and_max_not_specified:
            if kwargs['width'] == 16:
                kwargs['min_value'] = torch.finfo(torch.float16).min
                kwargs['max_value'] = torch.finfo(torch.float16).max
            elif kwargs['width'] == 32:
                kwargs['min_value'] = torch.finfo(torch.float32).min
                kwargs['max_value'] = torch.finfo(torch.float32).max
            elif kwargs['width'] == 64:
                kwargs['min_value'] = torch.finfo(torch.float64).min
                kwargs['max_value'] = torch.finfo(torch.float64).max
        kwargs.pop('width')
    return st.floats(*args, **kwargs)