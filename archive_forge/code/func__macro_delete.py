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
@as_subcommand_to('macro', 'delete', macro_delete_parser, help=macro_delete_help)
def _macro_delete(self, args: argparse.Namespace) -> None:
    """Delete macros"""
    self.last_result = True
    if args.all:
        self.macros.clear()
        self.poutput('All macros deleted')
    elif not args.names:
        self.perror('Either --all or macro name(s) must be specified')
        self.last_result = False
    else:
        for cur_name in utils.remove_duplicates(args.names):
            if cur_name in self.macros:
                del self.macros[cur_name]
                self.poutput(f"Macro '{cur_name}' deleted")
            else:
                self.perror(f"Macro '{cur_name}' does not exist")