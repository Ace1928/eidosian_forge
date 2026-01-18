from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.addresses import flags as addresses_flags
from googlecloudsdk.command_lib.util import completers
class PortRangesWithAll(object):
    """Particular keyword 'all' or a range of integer values."""

    def __init__(self, all_specified, ranges):
        self.all_specified = all_specified
        self.ranges = ranges

    @staticmethod
    def CreateParser():
        """Creates parser to parse keyword 'all' first before parse range."""

        def _Parse(string_value):
            if string_value.lower() == 'all':
                return PortRangesWithAll(True, [])
            else:
                type_parse = arg_parsers.ArgList(min_length=1, element_type=arg_parsers.Range.Parse)
                return PortRangesWithAll(False, type_parse(string_value))
        return _Parse