from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
@jit
def _xor_combine(a, b):
    return a ^ b