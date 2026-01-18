from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.functions import flags
from googlecloudsdk.command_lib.functions import secrets_config
from googlecloudsdk.command_lib.functions import util
from googlecloudsdk.command_lib.functions.v1.deploy import command as command_v1
from googlecloudsdk.command_lib.functions.v1.deploy import labels_util
from googlecloudsdk.command_lib.functions.v2.deploy import command as command_v2
from googlecloudsdk.command_lib.functions.v2.deploy import env_vars_util
from googlecloudsdk.command_lib.util.args import labels_util as args_labels_util
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class DeployAlpha(DeployBeta):
    """Create or update a Google Cloud Function."""

    @staticmethod
    def Args(parser):
        """Register alpha (and implicitly beta) flags for this command."""
        _CommonArgs(parser, base.ReleaseTrack.ALPHA)
        flags.AddBuildpackStackFlag(parser)