from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
@jit
def _argmax_combine_tie_break_left(value1, index1, value2, index2):
    return _argmax_combine(value1, index1, value2, index2, True)