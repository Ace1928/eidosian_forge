from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import locations as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.attached import resource_args
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.container.gkemulticloud import endpoint_util
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class GetServerConfig(base.Command):
    """Get Anthos Multi-Cloud server configuration for Attached clusters."""
    detailed_help = {'EXAMPLES': _EXAMPLES}

    @staticmethod
    def Args(parser):
        resource_args.AddLocationResourceArg(parser, 'to get server configuration')
        parser.display_info.AddFormat(constants.ATTACHED_SERVER_CONFIG_FORMAT)

    def Run(self, args):
        """Runs the get-server-config command."""
        location_ref = args.CONCEPTS.location.Parse()
        with endpoint_util.GkemulticloudEndpointOverride(location_ref.locationsId):
            log.status.Print('Fetching server config for {location}'.format(location=location_ref.locationsId))
            client = api_util.LocationsClient()
            return client.GetAttachedServerConfig(location_ref)