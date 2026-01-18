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
class TableElement(MarkdownElement):
    """MarkdownElement corresponding to `table_open`."""

    def __init__(self) -> None:
        self.header: TableHeaderElement | None = None
        self.body: TableBodyElement | None = None

    def on_child_close(self, context: 'MarkdownContext', child: 'MarkdownElement') -> bool:
        if isinstance(child, TableHeaderElement):
            self.header = child
        elif isinstance(child, TableBodyElement):
            self.body = child
        else:
            raise RuntimeError("Couldn't process markdown table.")
        return False

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        table = Table(box=box.SIMPLE_HEAVY)
        if self.header is not None and self.header.row is not None:
            for column in self.header.row.cells:
                table.add_column(column.content)
        if self.body is not None:
            for row in self.body.rows:
                row_content = [element.content for element in row.cells]
                table.add_row(*row_content)
        yield table