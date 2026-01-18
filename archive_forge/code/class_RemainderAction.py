import argparse
import arg_parsers
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import copy
import decimal
import json
import re
from dateutil import tz
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
from six.moves import zip  # pylint: disable=redefined-builtin
class RemainderAction(argparse._StoreAction):
    """An action with a couple of helpers to better handle --.

  argparse on its own does not properly handle -- implementation args.
  argparse.REMAINDER greedily steals valid flags before a --, and nargs='*' will
  bind to [] and not  parse args after --. This Action represents arguments to
  be passed through to a subcommand after --.

  Primarily, this Action provides two utility parsers to help a modified
  ArgumentParser parse -- properly.

  There is one additional property kwarg:
    example: A usage statement used to construct nice additional help.
  """

    def __init__(self, *args, **kwargs):
        if kwargs['nargs'] is not argparse.REMAINDER:
            raise ValueError('The RemainderAction should only be used when nargs=argparse.REMAINDER.')
        self.explanation = "The '--' argument must be specified between gcloud specific args on the left and {metavar} on the right.".format(metavar=kwargs['metavar'])
        if 'help' in kwargs:
            kwargs['help'] += '\n+\n' + self.explanation
            if 'example' in kwargs:
                kwargs['help'] += ' Example:\n\n' + kwargs['example']
                del kwargs['example']
        super(RemainderAction, self).__init__(*args, **kwargs)

    def _SplitOnDash(self, args):
        split_index = args.index('--')
        return (args[:split_index], args[split_index + 1:])

    def ParseKnownArgs(self, args, namespace):
        """Binds all args after -- to the namespace."""
        remainder_args = None
        if '--' in args:
            args, remainder_args = self._SplitOnDash(args)
        self(None, namespace, remainder_args)
        return (namespace, args)

    def ParseRemainingArgs(self, remaining_args, namespace, original_args):
        """Parses the unrecognized args from the end of the remaining_args.

    This method identifies all unrecognized arguments after the last argument
    recognized by a parser (but before --). It then either logs a warning and
    binds them to the namespace or raises an error, depending on strictness.

    Args:
      remaining_args: A list of arguments that the parsers did not recognize.
      namespace: The Namespace to bind to.
      original_args: The full list of arguments given to the top parser,

    Raises:
      ArgumentError: If there were remaining arguments after the last recognized
      argument and this action is strict.

    Returns:
      A tuple of the updated namespace and unrecognized arguments (before the
      last recognized argument).
    """
        if '--' in original_args:
            original_args, _ = self._SplitOnDash(original_args)
        split_index = 0
        for i, (arg1, arg2) in enumerate(zip(reversed(remaining_args), reversed(original_args))):
            if arg1 != arg2:
                split_index = len(remaining_args) - i
                break
        pass_through_args = remaining_args[split_index:]
        remaining_args = remaining_args[:split_index]
        if pass_through_args:
            msg = ('unrecognized args: {args}\n' + self.explanation).format(args=' '.join(pass_through_args))
            raise parser_errors.UnrecognizedArgumentsError(msg)
        self(None, namespace, pass_through_args)
        return (namespace, remaining_args)