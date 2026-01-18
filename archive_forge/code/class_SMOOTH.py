from __future__ import annotations
import functools
class SMOOTH(BuiltinFilter):
    name = 'Smooth'
    filterargs = ((3, 3), 13, 0, (1, 1, 1, 1, 5, 1, 1, 1, 1))