from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import versions_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml_engine import endpoint_util
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import region_util
from googlecloudsdk.command_lib.ml_engine import versions_util
@base.ReleaseTracks(base.ReleaseTrack.GA)
class SetDefault(base.DescribeCommand):
    """Sets an existing AI Platform version as the default for its model."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        _AddSetDefaultArgs(parser)

    def Run(self, args):
        return _Run(args)