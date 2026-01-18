from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
def AddUpstreamFleetFlags(self, with_destructive=False):
    """Adds upstream fleet flags."""
    if with_destructive:
        group = self.parser.add_mutually_exclusive_group()
        self._AddUpstreamFleetFlag(group)
        self._AddResetUpstreamFleetFlag(group)
    else:
        self._AddUpstreamFleetFlag(self.parser)