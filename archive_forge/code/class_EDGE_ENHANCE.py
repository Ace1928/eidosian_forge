from __future__ import annotations
import functools
class EDGE_ENHANCE(BuiltinFilter):
    name = 'Edge-enhance'
    filterargs = ((3, 3), 2, 0, (-1, -1, -1, -1, 10, -1, -1, -1, -1))