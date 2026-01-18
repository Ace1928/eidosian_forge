from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, argparse, contextlib
from . import completers, my_shlex as shlex
from .compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
from .completers import FilesCompleter, SuppressCompleter
from .my_argparse import IntrospectiveArgumentParser, action_is_satisfied, action_is_open, action_is_greedy
from .shellintegration import shellcode # noqa
def _patch_argument_parser(self):
    """
        Since argparse doesn't support much introspection, we monkey-patch it to replace the parse_known_args method and
        all actions with hooks that tell us which action was last taken or about to be taken, and let us have the parser
        figure out which subparsers need to be activated (then recursively monkey-patch those).
        We save all active ArgumentParsers to extract all their possible option names later.
        """
    self.active_parsers = []
    self.visited_positionals = []
    completer = self

    def patch(parser):
        completer.visited_positionals.append(parser)
        completer.active_parsers.append(parser)
        if isinstance(parser, IntrospectiveArgumentParser):
            return
        classname = 'MonkeyPatchedIntrospectiveArgumentParser'
        if USING_PYTHON2:
            classname = bytes(classname)
        parser.__class__ = type(classname, (IntrospectiveArgumentParser, parser.__class__), {})
        for action in parser._actions:
            if hasattr(action, '_orig_class'):
                continue

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
            action._orig_class = action.__class__
            action._orig_callable = action.__call__
            action.__class__ = IntrospectAction
    patch(self._parser)
    debug('Active parsers:', self.active_parsers)
    debug('Visited positionals:', self.visited_positionals)
    return self.active_parsers