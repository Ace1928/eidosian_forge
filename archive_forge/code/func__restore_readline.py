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
def _restore_readline(self, readline_settings: _SavedReadlineSettings) -> None:
    """
        Called at end of command loop to restore saved readline settings

        :param readline_settings: the readline settings to restore
        """
    if self._completion_supported():
        readline.set_completer(readline_settings.completer)
        readline.set_completer_delims(readline_settings.delims)
        if rl_type == RlType.GNU:
            readline.set_completion_display_matches_hook(None)
            rl_basic_quote_characters.value = readline_settings.basic_quotes
        elif rl_type == RlType.PYREADLINE:
            readline.rl.mode._display_completions = orig_pyreadline_display