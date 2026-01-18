import numpy as np
from numba import uint64, uint32, uint16, uint8
from numba.core.extending import register_jitable
from numba.np.random._constants import (UINT32_MAX, UINT64_MAX,
from numba.np.random.generator_core import next_uint32, next_uint64
@register_jitable
def _randint_arg_check(low, high, endpoint, lower_bound, upper_bound):
    """
    Check that low and high are within the bounds
    for the given datatype.
    """
    if low < lower_bound:
        raise ValueError('low is out of bounds')
    if high > 0:
        high = uint64(high)
        if not endpoint:
            high -= uint64(1)
        upper_bound = uint64(upper_bound)
        if low > 0:
            low = uint64(low)
        if high > upper_bound:
            raise ValueError('high is out of bounds')
        if low > high:
            raise ValueError('low is greater than high in given interval')
    else:
        if high > upper_bound:
            raise ValueError('high is out of bounds')
        if low > high:
            raise ValueError('low is greater than high in given interval')