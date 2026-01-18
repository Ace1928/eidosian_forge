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
def _restore_cmd2_env(self, cmd2_env: _SavedCmd2Env) -> None:
    """
        Restore cmd2 environment after exiting an interactive Python shell

        :param cmd2_env: the environment settings to restore
        """
    sys.stdout = cmd2_env.sys_stdout
    sys.stdin = cmd2_env.sys_stdin
    if rl_type != RlType.NONE:
        self._py_history.clear()
        for i in range(1, readline.get_current_history_length() + 1):
            self._py_history.append(readline.get_history_item(i))
        readline.clear_history()
        for item in cmd2_env.history:
            readline.add_history(item)
        if self._completion_supported():
            readline.set_completer(cmd2_env.readline_settings.completer)
            readline.set_completer_delims(cmd2_env.readline_settings.delims)
            if rl_type == RlType.GNU:
                rl_basic_quote_characters.value = cmd2_env.readline_settings.basic_quotes
                if 'gnureadline' in sys.modules:
                    if cmd2_env.readline_module is None:
                        del sys.modules['readline']
                    else:
                        sys.modules['readline'] = cmd2_env.readline_module