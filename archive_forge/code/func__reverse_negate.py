from fontTools.varLib.models import supportScalar
from fontTools.misc.fixedTools import MAX_F2DOT14
from functools import lru_cache
def _reverse_negate(v):
    return (-v[2], -v[1], -v[0])