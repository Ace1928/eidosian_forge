from __future__ import annotations
import math as _math  # unexport, see #17
import typing
from functools import partial as _partial
from functools import wraps as _wraps  # unexport, see #17
def _hpluv_to_rgb(_hx_tuple: Triplet) -> RGBColor:
    return lch_to_rgb(hpluv_to_lch(_hx_tuple))