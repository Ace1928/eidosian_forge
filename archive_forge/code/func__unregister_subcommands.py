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
def _unregister_subcommands(self, cmdset: Union[CommandSet, 'Cmd']) -> None:
    """
        Unregister subcommands from their base command

        :param cmdset: CommandSet containing subcommands
        """
    if not (cmdset is self or cmdset in self._installed_command_sets):
        raise CommandSetRegistrationError('Cannot unregister subcommands with an unregistered CommandSet')
    methods = inspect.getmembers(cmdset, predicate=lambda meth: isinstance(meth, Callable) and hasattr(meth, constants.SUBCMD_ATTR_NAME) and hasattr(meth, constants.SUBCMD_ATTR_COMMAND) and hasattr(meth, constants.CMD_ATTR_ARGPARSER))
    for method_name, method in methods:
        subcommand_name = getattr(method, constants.SUBCMD_ATTR_NAME)
        command_name = getattr(method, constants.SUBCMD_ATTR_COMMAND)
        if command_name in self.disabled_commands:
            command_func = self.disabled_commands[command_name].command_function
        else:
            command_func = self.cmd_func(command_name)
        if command_func is None:
            raise CommandSetRegistrationError(f"Could not find command '{command_name}' needed by subcommand: {str(method)}")
        command_parser = getattr(command_func, constants.CMD_ATTR_ARGPARSER, None)
        if command_parser is None:
            raise CommandSetRegistrationError(f"Could not find argparser for command '{command_name}' needed by subcommand: {str(method)}")
        for action in command_parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                action.remove_parser(subcommand_name)
                break