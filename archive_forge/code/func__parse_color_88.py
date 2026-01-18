from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def _parse_color_88(desc: str) -> int | None:
    """
    Return a color number for the description desc.
    'h0'..'h87' -> 0..87 actual color number
    '#000'..'#fff' -> 16..79 color cube colors
    'g0'..'g100' -> 16, 80..87, 79 grays and color cube black/white
    'g#00'..'g#ff' -> 16, 80...87, 79 gray and color cube black/white

    Returns None if desc is invalid.

    >>> _parse_color_88('h142')
    >>> _parse_color_88('h42')
    42
    >>> _parse_color_88('#f00')
    64
    >>> _parse_color_88('g100')
    79
    >>> _parse_color_88('g#80')
    83
    """
    if len(desc) == 7:
        desc = desc[0:2] + desc[3] + desc[5]
    if len(desc) > 4:
        return None
    try:
        if desc.startswith('h'):
            num = int(desc[1:], 10)
            if num < 0 or num > 87:
                return None
            return num
        if desc.startswith('#') and len(desc) == 4:
            rgb = int(desc[1:], 16)
            if rgb < 0:
                return None
            b, rgb = (rgb % 16, rgb // 16)
            g, r = (rgb % 16, rgb // 16)
            r = _CUBE_88_LOOKUP_16[r]
            g = _CUBE_88_LOOKUP_16[g]
            b = _CUBE_88_LOOKUP_16[b]
            return _CUBE_START + (r * _CUBE_SIZE_88 + g) * _CUBE_SIZE_88 + b
        if desc.startswith('g#'):
            gray = int(desc[2:], 16)
            if gray < 0 or gray > 255:
                return None
            gray = _GRAY_88_LOOKUP[gray]
        elif desc.startswith('g'):
            gray = int(desc[1:], 10)
            if gray < 0 or gray > 100:
                return None
            gray = _GRAY_88_LOOKUP_101[gray]
        else:
            return None
        if gray == 0:
            return _CUBE_BLACK
        gray -= 1
        if gray == _GRAY_SIZE_88:
            return _CUBE_WHITE_88
        return _GRAY_START_88 + gray
    except ValueError:
        return None