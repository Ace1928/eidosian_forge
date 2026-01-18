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
def _complete_flags(self, text: str, line: str, begidx: int, endidx: int, matched_flags: List[str]) -> List[str]:
    """Tab completion routine for a parsers unused flags"""
    match_against = []
    for flag in self._flags:
        if flag not in matched_flags:
            action = self._flag_to_action[flag]
            if action.help != argparse.SUPPRESS:
                match_against.append(flag)
    matches = self._cmd2_app.basic_complete(text, line, begidx, endidx, match_against)
    matched_actions: Dict[argparse.Action, List[str]] = dict()
    for flag in matches:
        action = self._flag_to_action[flag]
        matched_actions.setdefault(action, [])
        matched_actions[action].append(flag)
    for action, option_strings in matched_actions.items():
        flag_text = ', '.join(option_strings)
        if not action.required:
            flag_text = '[' + flag_text + ']'
        self._cmd2_app.display_matches.append(flag_text)
    return matches