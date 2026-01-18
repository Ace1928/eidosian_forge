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
def _display_matches_pyreadline(self, matches: List[str]) -> None:
    """Prints a match list using pyreadline3's _display_completions()

        :param matches: the tab completion matches to display
        """
    if rl_type == RlType.PYREADLINE:
        hint_printed = False
        if self.always_show_hint and self.completion_hint:
            hint_printed = True
            readline.rl.mode.console.write('\n' + self.completion_hint)
        if self.formatted_completions:
            if not hint_printed:
                readline.rl.mode.console.write('\n')
            readline.rl.mode.console.write('\n' + self.formatted_completions + '\n\n')
            rl_force_redisplay()
        else:
            if self.display_matches:
                matches_to_display = self.display_matches
            else:
                matches_to_display = matches
            matches_to_display, _ = self._pad_matches_to_display(matches_to_display)
            orig_pyreadline_display(matches_to_display)