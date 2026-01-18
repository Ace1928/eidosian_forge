from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.model_monitoring_jobs import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ResumeGa(base.SilentCommand):
    """Resume a paused Vertex AI model deployment monitoring job."""

    @staticmethod
    def Args(parser):
        flags.AddModelMonitoringJobResourceArg(parser, 'to resume')

    def Run(self, args):
        return _Run(args, constants.GA_VERSION, self.ReleaseTrack().prefix)