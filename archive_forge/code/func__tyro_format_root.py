from __future__ import annotations
import argparse
import contextlib
import dataclasses
import difflib
import itertools
import re as _re
import shlex
import shutil
import sys
from gettext import gettext as _
from typing import Any, Dict, Generator, Iterable, List, NoReturn, Optional, Set, Tuple
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from typing_extensions import override
from . import _arguments, _strings, conf
from ._parsers import ParserSpecification
def _tyro_format_root(self):
    console = Console(width=self.formatter._width, theme=THEME.as_rich_theme())
    with console.capture() as capture:
        top_parts = []
        column_parts = []
        column_parts_lines = []
        for func, args in self.items:
            item_content = func(*args)
            if item_content is None:
                pass
            elif isinstance(item_content, str):
                if item_content.strip() == '':
                    continue
                top_parts.append(Text.from_ansi(item_content))
            else:
                assert isinstance(item_content, Panel)
                column_parts.append(item_content)
                column_parts_lines.append(str_from_rich(item_content, width=65).strip().count('\n') + 1)
        min_column_width = 65
        height_breakpoint = 50
        column_count = max(1, min(sum(column_parts_lines) // height_breakpoint + 1, self.formatter._width // min_column_width, len(column_parts)))
        if column_count > 1:
            column_width = self.formatter._width // column_count - 1
            column_parts_lines = map(lambda p: str_from_rich(p, width=column_width).strip().count('\n') + 1, column_parts)
        else:
            column_width = None
        column_lines = [0 for i in range(column_count)]
        column_parts_grouped = [[] for i in range(column_count)]
        for p, l in zip(column_parts, column_parts_lines):
            chosen_column = column_lines.index(min(column_lines))
            column_parts_grouped[chosen_column].append(p)
            column_lines[chosen_column] += l
        columns = Columns([Group(*g) for g in column_parts_grouped], column_first=True, width=column_width)
        console.print(Group(*top_parts))
        console.print(columns)
    return capture.get()