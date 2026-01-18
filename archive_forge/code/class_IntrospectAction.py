from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, argparse, contextlib
from . import completers, my_shlex as shlex
from .compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
from .completers import FilesCompleter, SuppressCompleter
from .my_argparse import IntrospectiveArgumentParser, action_is_satisfied, action_is_open, action_is_greedy
from .shellintegration import shellcode # noqa
class IntrospectAction(action.__class__):

    def __call__(self, parser, namespace, values, option_string=None):
        debug('Action stub called on', self)
        debug('\targs:', parser, namespace, values, option_string)
        debug('\torig class:', self._orig_class)
        debug('\torig callable:', self._orig_callable)
        if not completer.completing:
            self._orig_callable(parser, namespace, values, option_string=option_string)
        elif issubclass(self._orig_class, argparse._SubParsersAction):
            debug('orig class is a subparsers action: patching and running it')
            patch(self._name_parser_map[values[0]])
            self._orig_callable(parser, namespace, values, option_string=option_string)
        elif self._orig_class in safe_actions:
            if not self.option_strings:
                completer.visited_positionals.append(self)
            self._orig_callable(parser, namespace, values, option_string=option_string)