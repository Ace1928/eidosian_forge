from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def __set_foreground(self, foreground: str) -> None:
    color = None
    flags = 0
    for part in foreground.split(','):
        part = part.strip()
        if part in _ATTRIBUTES:
            if flags & _ATTRIBUTES[part]:
                raise AttrSpecError(f'Setting {part!r} specified more than once in foreground ({foreground!r})')
            flags |= _ATTRIBUTES[part]
            continue
        if part in {'', 'default'}:
            scolor = 0
        elif part in _BASIC_COLORS:
            scolor = _BASIC_COLORS.index(part)
            flags |= _FG_BASIC_COLOR
        elif self.__value & _HIGH_88_COLOR:
            scolor = _parse_color_88(part)
            flags |= _FG_HIGH_COLOR
        elif self.__value & _HIGH_TRUE_COLOR:
            scolor = _parse_color_true(part)
            flags |= _FG_TRUE_COLOR
        else:
            scolor = _parse_color_256(_true_to_256(part) or part)
            flags |= _FG_HIGH_COLOR
        if scolor is None:
            raise AttrSpecError(f'Unrecognised color specification {part!r} in foreground ({foreground!r})')
        if color is not None:
            raise AttrSpecError(f'More than one color given for foreground ({foreground!r})')
        color = scolor
    if color is None:
        color = 0
    self.__value = self.__value & ~_FG_MASK | color | flags