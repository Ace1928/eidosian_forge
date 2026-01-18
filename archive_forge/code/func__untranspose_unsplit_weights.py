import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def _untranspose_unsplit_weights(params):
    Wx, Wh, bias = params
    xp = get_array_module(Wx)
    nO = Wh.shape[1]
    nI = Wx.shape[1]
    Wx = Wx.reshape((-1, 4, nI)).transpose((1, 0, 2)).reshape((-1, nI))
    Wh = Wh.reshape((-1, 4, nO)).transpose((1, 0, 2)).reshape((-1, nO))
    bias = bias.reshape((-1, 4)).transpose((1, 0)).reshape((-1,))
    zeros = xp.zeros(bias.shape, dtype='f')
    return xp.concatenate((Wx.ravel(), bias, Wh.ravel(), zeros))