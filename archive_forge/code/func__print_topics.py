import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
def _print_topics(self, header: str, cmds: List[str], verbose: bool) -> None:
    """Customized version of print_topics that can switch between verbose or traditional output"""
    import io
    if cmds:
        if not verbose:
            self.print_topics(header, cmds, 15, 80)
        else:
            widest = max([ansi.style_aware_wcswidth(command) for command in cmds])
            name_column = Column('', width=max(widest, 20))
            desc_column = Column('', width=80)
            topic_table = SimpleTable([name_column, desc_column], divider_char=self.ruler)
            table_str_buf = io.StringIO()
            if header:
                table_str_buf.write(header + '\n')
            divider = topic_table.generate_divider()
            if divider:
                table_str_buf.write(divider + '\n')
            topics = self.get_help_topics()
            for command in cmds:
                cmd_func = self.cmd_func(command)
                doc: Optional[str]
                if not hasattr(cmd_func, constants.CMD_ATTR_ARGPARSER) and command in topics:
                    help_func = getattr(self, constants.HELP_FUNC_PREFIX + command)
                    result = io.StringIO()
                    with redirect_stdout(result):
                        stdout_orig = self.stdout
                        try:
                            self.stdout = cast(TextIO, result)
                            help_func()
                        finally:
                            self.stdout = stdout_orig
                    doc = result.getvalue()
                else:
                    doc = cmd_func.__doc__
                cmd_desc = strip_doc_annotations(doc) if doc else ''
                table_row = topic_table.generate_data_row([command, cmd_desc])
                table_str_buf.write(table_row + '\n')
            self.poutput(table_str_buf.getvalue())