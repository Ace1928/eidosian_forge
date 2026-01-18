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
def _reset_completion_defaults(self) -> None:
    """
        Resets tab completion settings
        Needs to be called each time readline runs tab completion
        """
    self.allow_appended_space = True
    self.allow_closing_quote = True
    self.completion_hint = ''
    self.formatted_completions = ''
    self.completion_matches = []
    self.display_matches = []
    self.matches_delimited = False
    self.matches_sorted = False
    if rl_type == RlType.GNU:
        readline.set_completion_display_matches_hook(self._display_matches_gnu_readline)
    elif rl_type == RlType.PYREADLINE:
        readline.rl.mode._display_completions = self._display_matches_pyreadline