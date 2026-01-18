from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.assured import endpoint_util
from googlecloudsdk.api_lib.assured import violations as apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.base import ReleaseTrack
from googlecloudsdk.command_lib.assured import flags
@base.ReleaseTracks(ReleaseTrack.GA, ReleaseTrack.BETA, ReleaseTrack.ALPHA)
class Acknowledge(base.UpdateCommand):
    """Acknowledge an existing Assured Workloads compliance violation."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        flags.AddAcknowledgeViolationsFlags(parser)

    def Run(self, args):
        """Run the acknowledge command."""
        violation_resource = args.CONCEPTS.violation.Parse()
        region = violation_resource.Parent().Parent().Name()
        violation = violation_resource.RelativeName()
        with endpoint_util.AssuredWorkloadsEndpointOverridesFromRegion(release_track=self.ReleaseTrack(), region=region):
            client = apis.ViolationsClient(release_track=self.ReleaseTrack())
            if self.ReleaseTrack() == ReleaseTrack.GA:
                return client.Acknowledge(name=violation, comment=args.comment)
            return client.Acknowledge(name=violation, comment=args.comment, acknowledge_type=args.acknowledge_type)