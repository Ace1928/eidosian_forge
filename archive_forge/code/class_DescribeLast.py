from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.workflows import workflows
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class DescribeLast(base.DescribeCommand):
    """Show metadata for the last cached workflow execution."""
    detailed_help = {'DESCRIPTION': '{description}', 'EXAMPLES': '        To show metadata for the last cached workflow execution, run:\n\n          $ {command}\n        '}

    def Run(self, args):
        api_version = workflows.ReleaseTrackToApiVersion(self.ReleaseTrack())
        client = workflows.WorkflowExecutionClient(api_version)
        return client.Get(None)