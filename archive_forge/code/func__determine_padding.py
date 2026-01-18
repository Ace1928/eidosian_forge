import math
import os
import cupy
import numpy as np
from ._util import _get_inttype
from ._pba_2d import (_check_distances, _check_indices,
def _determine_padding(shape, block_size, m1, m2, m3, blockx, blocky):
    LCM = lcm(block_size, m1, m2, m3, blockx, blocky)
    orig_sz, orig_sy, orig_sx = shape
    round_up = False
    if orig_sx % LCM != 0:
        round_up = True
        sx = LCM * math.ceil(orig_sx / LCM)
    else:
        sx = orig_sx
    if orig_sy % LCM != 0:
        round_up = True
        sy = LCM * math.ceil(orig_sy / LCM)
    else:
        sy = orig_sy
    if orig_sz % LCM != 0:
        round_up = True
        sz = LCM * math.ceil(orig_sz / LCM)
    else:
        sz = orig_sz
    aniso = not sx == sy == sz
    if aniso or round_up:
        smax = max(sz, sy, sx)
        padding_width = ((0, smax - orig_sz), (0, smax - orig_sy), (0, smax - orig_sx))
    else:
        padding_width = None
    return padding_width