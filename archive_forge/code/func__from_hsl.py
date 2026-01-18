from __future__ import annotations
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence
@classmethod
def _from_hsl(cls, h: ParseableFloat, s: ParseableFloat, light: ParseableFloat, a: ParseableFloat=1) -> Color:
    h = float(h) / 360
    s = float(s) / 100
    _l = float(light) / 100
    if s == 0:
        r = _l
        g = r
        b = r
    else:
        luminocity2 = _l * (1 + s) if _l < 0.5 else _l + s - _l * s
        luminocity1 = 2 * _l - luminocity2

        def hue_to_rgb(lum1: float, lum2: float, hue: float) -> float:
            if hue < 0.0:
                hue += 1
            if hue > 1.0:
                hue -= 1
            if hue < 1.0 / 6.0:
                return lum1 + (lum2 - lum1) * 6.0 * hue
            if hue < 1.0 / 2.0:
                return lum2
            if hue < 2.0 / 3.0:
                return lum1 + (lum2 - lum1) * (2.0 / 3.0 - hue) * 6.0
            return lum1
        r = hue_to_rgb(luminocity1, luminocity2, h + 1.0 / 3.0)
        g = hue_to_rgb(luminocity1, luminocity2, h)
        b = hue_to_rgb(luminocity1, luminocity2, h - 1.0 / 3.0)
    return cls(round(r * 255), round(g * 255), round(b * 255), a)