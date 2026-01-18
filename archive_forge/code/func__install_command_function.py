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
def _install_command_function(self, command: str, command_wrapper: Callable[..., Any], context: str='') -> None:
    cmd_func_name = COMMAND_FUNC_PREFIX + command
    if hasattr(self, cmd_func_name):
        raise CommandSetRegistrationError(f'Attribute already exists: {cmd_func_name} ({context})')
    valid, errmsg = self.statement_parser.is_valid_command(command)
    if not valid:
        raise CommandSetRegistrationError(f"Invalid command name '{command}': {errmsg}")
    if command in self.aliases:
        self.pwarning(f"Deleting alias '{command}' because it shares its name with a new command")
        del self.aliases[command]
    if command in self.macros:
        self.pwarning(f"Deleting macro '{command}' because it shares its name with a new command")
        del self.macros[command]
    setattr(self, cmd_func_name, command_wrapper)