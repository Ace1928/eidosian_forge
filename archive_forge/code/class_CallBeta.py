from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.functions import flags
from googlecloudsdk.command_lib.functions import util
from googlecloudsdk.command_lib.functions.v1.call import command as command_v1
from googlecloudsdk.command_lib.functions.v2.call import command as command_v2
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class CallBeta(Call):
    """Triggers execution of a Google Cloud Function."""

    @staticmethod
    def Args(parser):
        """Register beta (and implicitly alpha) flags for this command."""
        Call.Args(parser)
        flags.AddGcloudHttpTimeoutFlag(parser)