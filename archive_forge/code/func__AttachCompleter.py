from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import display_info
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core.cache import completion_cache
def _AttachCompleter(self, arg, completer, positional):
    """Attaches a completer to arg if one is specified.

    Args:
      arg: The argument to attach the completer to.
      completer: The completer Completer class or argcomplete function object.
      positional: True if argument is a positional.
    """
    from googlecloudsdk.calliope import parser_completer
    if not completer:
        return
    if isinstance(completer, type):
        if positional and issubclass(completer, completion_cache.Completer):
            self.data.positional_completers.add(completer)
        arg.completer = parser_completer.ArgumentCompleter(completer, argument=arg)
    else:
        arg.completer = completer