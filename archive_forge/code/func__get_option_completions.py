from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, argparse, contextlib
from . import completers, my_shlex as shlex
from .compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
from .completers import FilesCompleter, SuppressCompleter
from .my_argparse import IntrospectiveArgumentParser, action_is_satisfied, action_is_open, action_is_greedy
from .shellintegration import shellcode # noqa
def _get_option_completions(self, parser, cword_prefix):
    self._display_completions.update([[' '.join((ensure_str(x) for x in action.option_strings if ensure_str(x).startswith(cword_prefix))), action.help] for action in parser._actions if action.option_strings])
    option_completions = []
    for action in parser._actions:
        if not self.print_suppressed:
            completer = getattr(action, 'completer', None)
            if isinstance(completer, SuppressCompleter) and completer.suppress():
                continue
            if action.help == argparse.SUPPRESS:
                continue
        if not self._action_allowed(action, parser):
            continue
        if not isinstance(action, argparse._SubParsersAction):
            option_completions += self._include_options(action, cword_prefix)
    return option_completions