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
@as_subcommand_to('macro', 'list', macro_list_parser, help=macro_list_help)
def _macro_list(self, args: argparse.Namespace) -> None:
    """List some or all macros as 'macro create' commands"""
    self.last_result = {}
    tokens_to_quote = constants.REDIRECTION_TOKENS
    tokens_to_quote.extend(self.statement_parser.terminators)
    if args.names:
        to_list = utils.remove_duplicates(args.names)
    else:
        to_list = sorted(self.macros, key=self.default_sort_key)
    not_found: List[str] = []
    for name in to_list:
        if name not in self.macros:
            not_found.append(name)
            continue
        tokens = shlex_split(self.macros[name].value)
        command = tokens[0]
        command_args = tokens[1:]
        utils.quote_specific_tokens(command_args, tokens_to_quote)
        val = command
        if command_args:
            val += ' ' + ' '.join(command_args)
        self.poutput(f'macro create {name} {val}')
        self.last_result[name] = val
    for name in not_found:
        self.perror(f"Macro '{name}' not found")