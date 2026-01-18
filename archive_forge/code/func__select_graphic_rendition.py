from __future__ import annotations
from string import Formatter
from typing import Generator
from prompt_toolkit.output.vt100 import BG_ANSI_COLORS, FG_ANSI_COLORS
from prompt_toolkit.output.vt100 import _256_colors as _256_colors_table
from .base import StyleAndTextTuples
def _select_graphic_rendition(self, attrs: list[int]) -> None:
    """
        Taken a list of graphics attributes and apply changes.
        """
    if not attrs:
        attrs = [0]
    else:
        attrs = list(attrs[::-1])
    while attrs:
        attr = attrs.pop()
        if attr in _fg_colors:
            self._color = _fg_colors[attr]
        elif attr in _bg_colors:
            self._bgcolor = _bg_colors[attr]
        elif attr == 1:
            self._bold = True
        elif attr == 3:
            self._italic = True
        elif attr == 4:
            self._underline = True
        elif attr == 5:
            self._blink = True
        elif attr == 6:
            self._blink = True
        elif attr == 7:
            self._reverse = True
        elif attr == 8:
            self._hidden = True
        elif attr == 9:
            self._strike = True
        elif attr == 22:
            self._bold = False
        elif attr == 23:
            self._italic = False
        elif attr == 24:
            self._underline = False
        elif attr == 25:
            self._blink = False
        elif attr == 27:
            self._reverse = False
        elif attr == 28:
            self._hidden = False
        elif attr == 29:
            self._strike = False
        elif not attr:
            self._color = None
            self._bgcolor = None
            self._bold = False
            self._underline = False
            self._strike = False
            self._italic = False
            self._blink = False
            self._reverse = False
            self._hidden = False
        elif attr in (38, 48) and len(attrs) > 1:
            n = attrs.pop()
            if n == 5 and len(attrs) >= 1:
                if attr == 38:
                    m = attrs.pop()
                    self._color = _256_colors.get(m)
                elif attr == 48:
                    m = attrs.pop()
                    self._bgcolor = _256_colors.get(m)
            if n == 2 and len(attrs) >= 3:
                try:
                    color_str = '#{:02x}{:02x}{:02x}'.format(attrs.pop(), attrs.pop(), attrs.pop())
                except IndexError:
                    pass
                else:
                    if attr == 38:
                        self._color = color_str
                    elif attr == 48:
                        self._bgcolor = color_str