from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.assured import endpoint_util
from googlecloudsdk.api_lib.assured import workloads as apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.base import ReleaseTrack
from googlecloudsdk.command_lib.assured import flags
@base.ReleaseTracks(ReleaseTrack.GA, ReleaseTrack.BETA, ReleaseTrack.ALPHA)
class EnableResourceMonitoring(base.UpdateCommand):
    """Enables Resource Monitoring for an Assured Workloads environment."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        flags.AddEnableResourceMonitoringFlags(parser)

    def Run(self, args):
        """Run the enables resource monitoring  command."""
        workload_resource = args.CONCEPTS.workload.Parse()
        region = workload_resource.Parent().Name()
        workload = workload_resource.RelativeName()
        with endpoint_util.AssuredWorkloadsEndpointOverridesFromRegion(release_track=self.ReleaseTrack(), region=region):
            client = apis.WorkloadsClient(release_track=self.ReleaseTrack())
            self.resource_name = workload
            return client.EnableResourceMonitoring(name=self.resource_name)