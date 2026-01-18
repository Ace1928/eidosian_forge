import os.path
import platform
import re
import sys
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
from pip._vendor.pygments.lexer import Lexer
from pip._vendor.pygments.lexers import get_lexer_by_name, guess_lexer_for_filename
from pip._vendor.pygments.style import Style as PygmentsStyle
from pip._vendor.pygments.styles import get_style_by_name
from pip._vendor.pygments.token import (
from pip._vendor.pygments.util import ClassNotFound
from pip._vendor.rich.containers import Lines
from pip._vendor.rich.padding import Padding, PaddingDimensions
from ._loop import loop_first
from .cells import cell_len
from .color import Color, blend_rgb
from .console import Console, ConsoleOptions, JustifyMethod, RenderResult
from .jupyter import JupyterMixin
from .measure import Measurement
from .segment import Segment, Segments
from .style import Style, StyleType
from .text import Text
def _get_number_styles(self, console: Console) -> Tuple[Style, Style, Style]:
    """Get background, number, and highlight styles for line numbers."""
    background_style = self._get_base_style()
    if background_style.transparent_background:
        return (Style.null(), Style(dim=True), Style.null())
    if console.color_system in ('256', 'truecolor'):
        number_style = Style.chain(background_style, self._theme.get_style_for_token(Token.Text), Style(color=self._get_line_numbers_color()), self.background_style)
        highlight_number_style = Style.chain(background_style, self._theme.get_style_for_token(Token.Text), Style(bold=True, color=self._get_line_numbers_color(0.9)), self.background_style)
    else:
        number_style = background_style + Style(dim=True)
        highlight_number_style = background_style + Style(dim=False)
    return (background_style, number_style, highlight_number_style)