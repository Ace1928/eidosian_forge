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
class TableRowElement(MarkdownElement):
    """MarkdownElement corresponding to `tr_open` and `tr_close`."""

    def __init__(self) -> None:
        self.cells: List[TableDataElement] = []

    def on_child_close(self, context: 'MarkdownContext', child: 'MarkdownElement') -> bool:
        assert isinstance(child, TableDataElement)
        self.cells.append(child)
        return False