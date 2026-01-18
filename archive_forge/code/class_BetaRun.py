from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.workflows import cache
from googlecloudsdk.api_lib.workflows import workflows
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.workflows import flags
from googlecloudsdk.command_lib.workflows import hooks
from googlecloudsdk.core import resources
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class BetaRun(Run):
    """Execute a workflow and wait for the execution to complete."""
    detailed_help = {'DESCRIPTION': '{description}', 'EXAMPLES': '        To execute a workflow named my-workflow with the data that will be passed to the workflow, run:\n\n          $ {command} my-workflow --data=my-data\n        '}

    @staticmethod
    def Args(parser):
        Run.CommonArgs(parser)
        flags.AddBetaLoggingArg(parser)

    def CallLogLevel(self, args):
        return args.call_log_level

    def Labels(self, args):
        return None

    def OverflowBufferingDisabled(self, args):
        return False