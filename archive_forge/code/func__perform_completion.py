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
def _perform_completion(self, text: str, line: str, begidx: int, endidx: int, custom_settings: Optional[utils.CustomCompletionSettings]=None) -> None:
    """
        Helper function for complete() that performs the actual completion

        :param text: the string prefix we are attempting to match (all matches must begin with it)
        :param line: the current input line with leading whitespace removed
        :param begidx: the beginning index of the prefix text
        :param endidx: the ending index of the prefix text
        :param custom_settings: optional prepopulated completion settings
        """
    command = ''
    if custom_settings is None:
        statement = self.statement_parser.parse_command_only(line)
        command = statement.command
        if not command:
            return
        expanded_line = statement.command_and_args
        rstripped_len = len(line) - len(line.rstrip())
        expanded_line += ' ' * rstripped_len
        if len(expanded_line) != len(line):
            diff = len(expanded_line) - len(line)
            begidx += diff
            endidx += diff
        line = expanded_line
    tokens, raw_tokens = self.tokens_for_completion(line, begidx, endidx)
    if not tokens:
        return
    if custom_settings is None:
        if command in self.macros:
            completer_func = self.path_complete
        elif command in self.get_all_commands():
            func_attr = getattr(self, constants.COMPLETER_FUNC_PREFIX + command, None)
            if func_attr is not None:
                completer_func = func_attr
            else:
                func = self.cmd_func(command)
                argparser: Optional[argparse.ArgumentParser] = getattr(func, constants.CMD_ATTR_ARGPARSER, None)
                if func is not None and argparser is not None:
                    preserve_quotes = getattr(func, constants.CMD_ATTR_PRESERVE_QUOTES)
                    cmd_set = self._cmd_to_command_sets[command] if command in self._cmd_to_command_sets else None
                    completer_type = self._determine_ap_completer_type(argparser)
                    completer = completer_type(argparser, self)
                    completer_func = functools.partial(completer.complete, tokens=raw_tokens[1:] if preserve_quotes else tokens[1:], cmd_set=cmd_set)
                else:
                    completer_func = self.completedefault
        elif self.default_to_shell and command in utils.get_exes_in_path(command):
            completer_func = self.path_complete
        else:
            completer_func = self.completedefault
    else:
        completer_type = self._determine_ap_completer_type(custom_settings.parser)
        completer = completer_type(custom_settings.parser, self)
        completer_func = functools.partial(completer.complete, tokens=raw_tokens if custom_settings.preserve_quotes else tokens, cmd_set=None)
    text_to_remove = ''
    raw_completion_token = raw_tokens[-1]
    completion_token_quote = ''
    if raw_completion_token and raw_completion_token[0] in constants.QUOTES:
        completion_token_quote = raw_completion_token[0]
        actual_begidx = line[:endidx].rfind(tokens[-1])
        if actual_begidx != begidx:
            text_to_remove = line[actual_begidx:begidx]
            text = text_to_remove + text
            begidx = actual_begidx
    self.completion_matches = self._redirect_complete(text, line, begidx, endidx, completer_func)
    if self.completion_matches:
        self.completion_matches = utils.remove_duplicates(self.completion_matches)
        self.display_matches = utils.remove_duplicates(self.display_matches)
        if not self.display_matches:
            import copy
            self.display_matches = copy.copy(self.completion_matches)
        if not completion_token_quote:
            add_quote = False
            common_prefix = os.path.commonprefix(self.completion_matches)
            if self.matches_delimited:
                display_prefix = os.path.commonprefix(self.display_matches)
                if ' ' in common_prefix or (display_prefix and any((' ' in match for match in self.display_matches))):
                    add_quote = True
            elif common_prefix and any((' ' in match for match in self.completion_matches)):
                add_quote = True
            if add_quote:
                if any(('"' in match for match in self.completion_matches)):
                    completion_token_quote = "'"
                else:
                    completion_token_quote = '"'
                self.completion_matches = [completion_token_quote + match for match in self.completion_matches]
        elif text_to_remove:
            self.completion_matches = [match.replace(text_to_remove, '', 1) for match in self.completion_matches]
        if len(self.completion_matches) == 1 and self.allow_closing_quote and completion_token_quote:
            self.completion_matches[0] += completion_token_quote