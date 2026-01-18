from __future__ import annotations
from string import Formatter
from typing import Generator
from prompt_toolkit.output.vt100 import BG_ANSI_COLORS, FG_ANSI_COLORS
from prompt_toolkit.output.vt100 import _256_colors as _256_colors_table
from .base import StyleAndTextTuples
def _create_style_string(self) -> str:
    """
        Turn current style flags into a string for usage in a formatted text.
        """
    result = []
    if self._color:
        result.append(self._color)
    if self._bgcolor:
        result.append('bg:' + self._bgcolor)
    if self._bold:
        result.append('bold')
    if self._underline:
        result.append('underline')
    if self._strike:
        result.append('strike')
    if self._italic:
        result.append('italic')
    if self._blink:
        result.append('blink')
    if self._reverse:
        result.append('reverse')
    if self._hidden:
        result.append('hidden')
    return ' '.join(result)