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
def _resolve_func_self(self, cmd_support_func: Callable[..., Any], cmd_self: Union[CommandSet, 'Cmd', None]) -> Optional[object]:
    """
        Attempt to resolve a candidate instance to pass as 'self' for an unbound class method that was
        used when defining command's argparse object. Since we restrict registration to only a single CommandSet
        instance of each type, using type is a reasonably safe way to resolve the correct object instance

        :param cmd_support_func: command support function. This could be a completer or namespace provider
        :param cmd_self: The `self` associated with the command or subcommand
        """
    func_class: Optional[Type[Any]] = get_defining_class(cmd_support_func)
    if func_class is not None and issubclass(func_class, CommandSet):
        func_self: Optional[Union[CommandSet, 'Cmd']]
        if isinstance(cmd_self, func_class):
            func_self = cmd_self
        else:
            func_self = None
            candidate_sets: List[CommandSet] = []
            for installed_cmd_set in self._installed_command_sets:
                if type(installed_cmd_set) == func_class:
                    func_self = installed_cmd_set
                    break
                if isinstance(installed_cmd_set, func_class):
                    candidate_sets.append(installed_cmd_set)
            if func_self is None and len(candidate_sets) == 1:
                func_self = candidate_sets[0]
        return func_self
    else:
        return self