import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def _transpose_weights(params):
    (Wx, bx), (Wh, bh) = params
    xp = get_array_module(Wx)
    Wx = Wx.reshape((4, -1, Wx.shape[-1]))
    Wx = Wx.transpose((1, 0, 2)).reshape((-1, Wx.shape[-1]))
    bx = bx.reshape((4, -1)).transpose((1, 0)).reshape((-1,))
    Wh = Wh.reshape((4, -1, Wh.shape[-1]))
    Wh = Wh.transpose((1, 0, 2)).reshape((-1, Wh.shape[-1]))
    bh = bh.reshape((4, -1)).transpose((1, 0)).reshape((-1,))
    ascontig = xp.ascontiguousarray
    Wx = ascontig(Wx)
    Wh = ascontig(Wh)
    bias = ascontig(bx) + bh
    return (Wx, Wh, bias)