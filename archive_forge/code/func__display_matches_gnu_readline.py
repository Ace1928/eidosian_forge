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
def _display_matches_gnu_readline(self, substitution: str, matches: List[str], longest_match_length: int) -> None:
    """Prints a match list using GNU readline's rl_display_match_list()

        :param substitution: the substitution written to the command line
        :param matches: the tab completion matches to display
        :param longest_match_length: longest printed length of the matches
        """
    if rl_type == RlType.GNU:
        hint_printed = False
        if self.always_show_hint and self.completion_hint:
            hint_printed = True
            sys.stdout.write('\n' + self.completion_hint)
        if self.formatted_completions:
            if not hint_printed:
                sys.stdout.write('\n')
            sys.stdout.write('\n' + self.formatted_completions + '\n\n')
        else:
            if self.display_matches:
                matches_to_display = self.display_matches
                longest_match_length = 0
                for cur_match in matches_to_display:
                    cur_length = ansi.style_aware_wcswidth(cur_match)
                    if cur_length > longest_match_length:
                        longest_match_length = cur_length
            else:
                matches_to_display = matches
            matches_to_display, padding_length = self._pad_matches_to_display(matches_to_display)
            longest_match_length += padding_length
            encoded_substitution = bytes(substitution, encoding='utf-8')
            encoded_matches = [bytes(cur_match, encoding='utf-8') for cur_match in matches_to_display]
            strings_array = cast(List[Optional[bytes]], (ctypes.c_char_p * (1 + len(encoded_matches) + 1))())
            strings_array[0] = encoded_substitution
            strings_array[1:-1] = encoded_matches
            strings_array[-1] = None
            readline_lib.rl_display_match_list(strings_array, len(encoded_matches), longest_match_length)
        rl_force_redisplay()