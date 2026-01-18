from __future__ import annotations
import itertools
import shutil
import os
import textwrap
import typing as T
import collections
from . import build
from . import coredata
from . import environment
from . import mesonlib
from . import mintro
from . import mlog
from .ast import AstIDGenerator, IntrospectionInterpreter
from .mesonlib import MachineChoice, OptionKey
def _add_line(self, name: LOGLINE, value: LOGLINE, choices: LOGLINE, descr: LOGLINE) -> None:
    if isinstance(name, mlog.AnsiDecorator):
        name.text = ' ' * self.print_margin + name.text
    else:
        name = ' ' * self.print_margin + name
    self.name_col.append(name)
    self.value_col.append(value)
    self.choices_col.append(choices)
    self.descr_col.append(descr)