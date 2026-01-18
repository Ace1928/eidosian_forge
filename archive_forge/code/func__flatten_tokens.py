from __future__ import annotations
import sys
from typing import ClassVar, Dict, Iterable, List, Optional, Type, Union
from markdown_it import MarkdownIt
from markdown_it.token import Token
from rich.table import Table
from . import box
from ._loop import loop_first
from ._stack import Stack
from .console import Console, ConsoleOptions, JustifyMethod, RenderResult
from .containers import Renderables
from .jupyter import JupyterMixin
from .panel import Panel
from .rule import Rule
from .segment import Segment
from .style import Style, StyleStack
from .syntax import Syntax
from .text import Text, TextType
def _flatten_tokens(self, tokens: Iterable[Token]) -> Iterable[Token]:
    """Flattens the token stream."""
    for token in tokens:
        is_fence = token.type == 'fence'
        is_image = token.tag == 'img'
        if token.children and (not (is_image or is_fence)):
            yield from self._flatten_tokens(token.children)
        else:
            yield token