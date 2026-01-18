from __future__ import annotations
import functools
class SHARPEN(BuiltinFilter):
    name = 'Sharpen'
    filterargs = ((3, 3), 16, 0, (-2, -2, -2, -2, 32, -2, -2, -2, -2))