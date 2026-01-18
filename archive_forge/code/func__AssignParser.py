from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import re
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.calliope import display
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.util import text
import six
def _AssignParser(self, parser_group, allow_positional_args):
    """Assign a parser group to model this Command or CommandGroup.

    Args:
      parser_group: argparse._ArgumentGroup, the group that will model this
          command or group's arguments.
      allow_positional_args: bool, Whether to allow positional args for this
          group or not.

    """
    if not parser_group:
        self._parser = parser_extensions.ArgumentParser(description=self.long_help, add_help=False, prog=self.dotted_name, calliope_command=self)
    else:
        self._parser = parser_group.add_parser(self.cli_name, help=self.short_help, description=self.long_help, add_help=False, prog=self.dotted_name, calliope_command=self)
    self._sub_parser = None
    self.ai = parser_arguments.ArgumentInterceptor(parser=self._parser, is_global=not parser_group, cli_generator=self._cli_generator, allow_positional=allow_positional_args)
    self.ai.add_argument('-h', action=actions.ShortHelpAction(self), is_replicated=True, category=base.COMMONLY_USED_FLAGS, help='Print a summary help and exit.')
    self.ai.add_argument('--help', action=actions.RenderDocumentAction(self, '--help'), is_replicated=True, category=base.COMMONLY_USED_FLAGS, help='Display detailed help.')
    self.ai.add_argument('--document', action=actions.RenderDocumentAction(self), is_replicated=True, nargs=1, metavar='ATTRIBUTES', type=arg_parsers.ArgDict(), hidden=True, help='THIS TEXT SHOULD BE HIDDEN')
    self._AcquireArgs()