import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from onnx import MapProto, OptionalProto, SequenceProto, TensorProto, helper, subbyte
from onnx.external_data_helper import load_external_data_for_tensor, uses_external_data
def _float8e5m2_to_float32_scalar(ival: int, fn: bool, uz: bool) -> np.float32:
    if fn and uz:
        if ival == 128:
            return np.float32(np.nan)
        exponent_bias = 16
    elif not fn and (not uz):
        if ival in {253, 254, 255}:
            return np.float32(-np.nan)
        if ival in {125, 126, 127}:
            return np.float32(np.nan)
        if ival == 252:
            return np.float32(-np.inf)
        if ival == 124:
            return np.float32(np.inf)
        exponent_bias = 15
    else:
        raise NotImplementedError('fn and uz must be both False or True.')
    expo = (ival & 124) >> 2
    mant = ival & 3
    sign = ival & 128
    res = sign << 24
    if expo == 0:
        if mant > 0:
            expo = 127 - exponent_bias
            if mant & 2 == 0:
                mant &= 1
                mant <<= 1
                expo -= 1
            res |= (mant & 1) << 22
            res |= expo << 23
    else:
        res |= mant << 21
        expo += 127 - exponent_bias
        res |= expo << 23
    f = np.uint32(res).view(np.float32)
    return f