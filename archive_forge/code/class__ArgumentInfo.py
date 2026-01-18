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
@dataclasses.dataclass(frozen=True)
class _ArgumentInfo:
    option_strings: Tuple[str, ...]
    metavar: Optional[str]
    usage_hint: str
    help: Optional[str]
    subcommand_match_score: float
    'Priority value used when an argument is in the current subcommand tree.'