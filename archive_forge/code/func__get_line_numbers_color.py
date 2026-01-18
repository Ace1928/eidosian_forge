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
def _get_line_numbers_color(self, blend: float=0.3) -> Color:
    background_style = self._theme.get_background_style() + self.background_style
    background_color = background_style.bgcolor
    if background_color is None or background_color.is_system_defined:
        return Color.default()
    foreground_color = self._get_token_color(Token.Text)
    if foreground_color is None or foreground_color.is_system_defined:
        return foreground_color or Color.default()
    new_color = blend_rgb(background_color.get_truecolor(), foreground_color.get_truecolor(), cross_fade=blend)
    return Color.from_triplet(new_color)