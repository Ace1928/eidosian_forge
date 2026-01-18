from __future__ import annotations
from string import Formatter
from typing import Generator
from prompt_toolkit.output.vt100 import BG_ANSI_COLORS, FG_ANSI_COLORS
from prompt_toolkit.output.vt100 import _256_colors as _256_colors_table
from .base import StyleAndTextTuples
def __pt_formatted_text__(self) -> StyleAndTextTuples:
    return self._formatted_text