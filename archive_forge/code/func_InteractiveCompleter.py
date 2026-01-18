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
def InteractiveCompleter(self, args):
    """Returns the interactive completion choices for args or None.

    Args:
      args: The CLI tree parsed command args.

    Returns:
      (choices, offset):
        choices - The list of completion strings or None.
        offset - The completion prefix offset.
    """
    if self.empty and args[-1].value:
        args = args[:]
        args.append(parser.ArgToken('', parser.ArgTokenType.UNKNOWN, None))
    completions = self.arg_cache.Lookup(args)
    if not completions:
        prefix = self.DoExecutableCompletions() and self.IsPrefixArg(args)
        if not self.event.completion_requested and (not prefix):
            return (None, None)
        command = [arg.value for arg in args]
        with Spinner(self.SetSpinner):
            completions = self.coshell.GetCompletions(command, prefix=prefix)
        self.debug.get.count()
        if not completions:
            return (None, None)
        self.arg_cache.Update(args, completions)
    else:
        self.debug.hit.count()
    last = args[-1].value
    offset = -len(last)
    if False and len(completions) == 1 and completions[0].startswith(last):
        return (completions, offset)
    chop = len(os.path.dirname(last))
    uri_sep = _URI_SEP
    uri_sep_index = completions[0].find(uri_sep)
    if uri_sep_index > 0:
        if not last:
            chop = uri_sep_index + len(uri_sep)
    result = []
    strip_trailing_slash = len(completions) != 1
    for c in completions:
        path_completion = self.MakePathCompletion(c, offset, chop, strip_trailing_slash)
        if path_completion:
            result.append(path_completion)
    return (result, None)