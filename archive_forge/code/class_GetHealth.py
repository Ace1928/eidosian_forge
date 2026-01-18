from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.notebooks import instances as instance_util
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.notebooks import flags
@base.ReleaseTracks(base.ReleaseTrack.GA)
class GetHealth(base.DescribeCommand):
    """Request for checking if a notebook instance is healthy."""

    @classmethod
    def Args(cls, parser):
        """Register flags for this command."""
        api_version = util.ApiVersionSelector(cls.ReleaseTrack())
        flags.AddGetHealthInstanceFlags(api_version, parser)

    def Run(self, args):
        release_track = self.ReleaseTrack()
        client = util.GetClient(release_track)
        messages = util.GetMessages(release_track)
        instance_service = client.projects_locations_instances
        result = instance_service.GetInstanceHealth(instance_util.CreateInstanceGetHealthRequest(args, messages))
        return result