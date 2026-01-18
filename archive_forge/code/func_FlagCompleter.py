from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import sys
import threading
import time
from googlecloudsdk.calliope import parser_completer
from googlecloudsdk.command_lib.interactive import parser
from googlecloudsdk.command_lib.meta import generate_cli_trees
from googlecloudsdk.core import module_util
from googlecloudsdk.core.console import console_attr
from prompt_toolkit import completion
import six
def FlagCompleter(self, args):
    """Returns the flag completion choices for args or None.

    Args:
      args: The CLI tree parsed command args.

    Returns:
      (choices, offset):
        choices - The list of completion strings or None.
        offset - The completion prefix offset.
    """
    arg = args[-1]
    if arg.token_type == parser.ArgTokenType.FLAG_ARG and args[-2].token_type == parser.ArgTokenType.FLAG and (not arg.value and self.last in (' ', '=') or (arg.value and (not self.empty))):
        flag = args[-2].tree
        return self.ArgCompleter(args, flag, arg.value)
    elif arg.token_type == parser.ArgTokenType.FLAG:
        if not self.empty:
            flags = {}
            for a in reversed(args):
                if a.tree and parser.LOOKUP_FLAGS in a.tree:
                    flags = a.tree[parser.LOOKUP_FLAGS]
                    break
            completions = [k for k, v in six.iteritems(flags) if k != arg.value and k.startswith(arg.value) and (not self.IsSuppressed(v))]
            if completions:
                completions.append(arg.value)
                return (completions, -len(arg.value))
        flag = arg.tree
        if flag.get(parser.LOOKUP_TYPE) != 'bool':
            completions, offset = self.ArgCompleter(args, flag, '')
            if not self.empty and self.last != '=':
                completions = [' ' + c for c in completions]
            return (completions, offset)
    elif arg.value.startswith('-'):
        return ([k for k, v in six.iteritems(arg.tree[parser.LOOKUP_FLAGS]) if k.startswith(arg.value) and (not self.IsSuppressed(v))], -len(arg.value))
    return (None, 0)