from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
class FlagGroup(BinaryCommandFlag):
    """Encapsulates multiple flags that are logically added together."""

    def __init__(self, first, second, *args):
        """Create a new flag group.

    At least two flags must be specified.

    Args:
      first: the first flag in the group
      second: the second flag in the group
      *args: additional flags in the group
    """
        super(FlagGroup, self).__init__()
        all_flags = [first, second]
        all_flags.extend(args)
        self._flags = all_flags

    def AddToParser(self, parser):
        for f in self._flags:
            f.AddToParser(parser)

    def FormatFlags(self, args):
        all_flags = []
        for f in self._flags:
            all_flags.extend(f.FormatFlags(args))
        return all_flags