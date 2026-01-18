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
@as_subcommand_to('alias', 'delete', alias_delete_parser, help=alias_delete_help)
def _alias_delete(self, args: argparse.Namespace) -> None:
    """Delete aliases"""
    self.last_result = True
    if args.all:
        self.aliases.clear()
        self.poutput('All aliases deleted')
    elif not args.names:
        self.perror('Either --all or alias name(s) must be specified')
        self.last_result = False
    else:
        for cur_name in utils.remove_duplicates(args.names):
            if cur_name in self.aliases:
                del self.aliases[cur_name]
                self.poutput(f"Alias '{cur_name}' deleted")
            else:
                self.perror(f"Alias '{cur_name}' does not exist")