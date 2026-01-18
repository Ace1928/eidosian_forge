from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, argparse, contextlib
from . import completers, my_shlex as shlex
from .compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
from .completers import FilesCompleter, SuppressCompleter
from .my_argparse import IntrospectiveArgumentParser, action_is_satisfied, action_is_open, action_is_greedy
from .shellintegration import shellcode # noqa
def _include_options(self, action, cword_prefix):
    if len(cword_prefix) > 0 or self.always_complete_options is True:
        return [ensure_str(opt) for opt in action.option_strings if ensure_str(opt).startswith(cword_prefix)]
    long_opts = [ensure_str(opt) for opt in action.option_strings if len(opt) > 2]
    short_opts = [ensure_str(opt) for opt in action.option_strings if len(opt) <= 2]
    if self.always_complete_options == 'long':
        return long_opts if long_opts else short_opts
    elif self.always_complete_options == 'short':
        return short_opts if short_opts else long_opts
    return []