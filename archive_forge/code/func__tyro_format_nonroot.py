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
def _tyro_format_nonroot(self):
    description_part = None
    item_parts = []
    for func, args in self.items:
        if getattr(func, '__func__', None) is TyroArgparseHelpFormatter._format_action:
            action, = args
            assert isinstance(action, argparse.Action)
            item_parts.extend(self._format_action(action))
        else:
            item_content = func(*args)
            assert isinstance(item_content, str)
            if item_content.strip() != '':
                assert description_part is None
                description_part = Text.from_ansi(item_content.strip() + '\n', style=THEME.description)
    if len(item_parts) == 0:
        return None
    if self.heading is not argparse.SUPPRESS and self.heading is not None:
        current_indent = self.formatter._current_indent
        heading = '%*s%s:\n' % (current_indent, '', self.heading)
        heading = heading.strip()[:-1]
    else:
        heading = ''
    lines = list(itertools.chain(*map(lambda p: _strings.strip_ansi_sequences(str_from_rich(p, width=self.formatter._width, soft_wrap=True)).rstrip().split('\n'), item_parts + [description_part] if description_part is not None else item_parts)))
    max_width = max(map(len, lines))
    if self.formatter._tyro_rule is None:
        self.formatter._tyro_rule = Text.from_ansi('─' * max_width, style=THEME.border, overflow='crop')
    elif len(self.formatter._tyro_rule._text[0]) < max_width:
        self.formatter._tyro_rule._text = ['─' * max_width]
    if description_part is not None:
        item_parts = [description_part, self.formatter._tyro_rule] + item_parts
    return Panel(Group(*item_parts), title=heading, title_align='left', border_style=THEME.border)