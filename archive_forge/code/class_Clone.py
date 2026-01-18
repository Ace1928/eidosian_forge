from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.command_lib.sql import instances as command_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA)
class Clone(base.CreateCommand):
    """Clones a Cloud SQL instance."""
    detailed_help = DETAILED_HELP

    @classmethod
    def Args(cls, parser):
        """Declare flag and positional arguments for the command parser."""
        AddBaseArgs(parser)
        parser.display_info.AddCacheUpdater(flags.InstanceCompleter)

    def Run(self, args):
        return RunBaseCloneCommand(args, self.ReleaseTrack())