import argparse
import inspect
import numbers
from collections import (
from typing import (
from .ansi import (
from .constants import (
from .argparse_custom import (
from .command_definition import (
from .exceptions import (
from .table_creator import (
class _ArgumentState:
    """Keeps state of an argument being parsed"""

    def __init__(self, arg_action: argparse.Action) -> None:
        self.action = arg_action
        self.min: Union[int, str]
        self.max: Union[float, int, str]
        self.count = 0
        self.is_remainder = self.action.nargs == argparse.REMAINDER
        nargs_range = self.action.get_nargs_range()
        if nargs_range is not None:
            self.min = nargs_range[0]
            self.max = nargs_range[1]
        elif self.action.nargs is None:
            self.min = 1
            self.max = 1
        elif self.action.nargs == argparse.OPTIONAL:
            self.min = 0
            self.max = 1
        elif self.action.nargs == argparse.ZERO_OR_MORE or self.action.nargs == argparse.REMAINDER:
            self.min = 0
            self.max = INFINITY
        elif self.action.nargs == argparse.ONE_OR_MORE:
            self.min = 1
            self.max = INFINITY
        else:
            self.min = self.action.nargs
            self.max = self.action.nargs