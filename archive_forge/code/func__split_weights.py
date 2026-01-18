import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def _split_weights(params: Floats1d, i: int, nO: int, nI: int, params_i: int):
    Wx_size = 4 * nO * nI
    bx_size = 4 * nO
    Wh_size = 4 * nO * nO
    bh_size = 4 * nO
    Wx = params[params_i:params_i + Wx_size].reshape((4 * nO, nI))
    params_i += Wx_size
    bx = params[params_i:params_i + bx_size].reshape((4 * nO,))
    params_i += bx_size
    Wh = params[params_i:params_i + Wh_size].reshape((4 * nO, nO))
    params_i += Wh_size
    bh = params[params_i:params_i + bh_size].reshape((4 * nO,))
    params_i += bh_size
    return (((Wx, bx), (Wh, bh)), params_i)