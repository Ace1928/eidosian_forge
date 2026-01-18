from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.network_connectivity import networkconnectivity_api
from googlecloudsdk.api_lib.network_connectivity import networkconnectivity_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.network_connectivity import flags
from googlecloudsdk.command_lib.network_connectivity import util
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ListSpokes(base.ListCommand):
    """List hub spokes.

  Retrieve and display a list of all spokes associated with a hub.
  """

    @staticmethod
    def Args(parser):
        base.URI_FLAG.RemoveFromParser(parser)
        flags.AddSpokeLocationsFlag(parser)
        flags.AddViewFlag(parser)
        flags.AddHubResourceArg(parser, 'associated with the returned list of\n                            spokes')
        parser.display_info.AddFormat(util.LIST_SPOKES_FORMAT)

    def Run(self, args):
        release_track = self.ReleaseTrack()
        view = ViewToEnum(args.view, release_track)
        client = networkconnectivity_api.HubsClient(release_track)
        hub_ref = args.CONCEPTS.hub.Parse()
        return client.ListHubSpokes(hub_ref, spoke_locations=args.spoke_locations, limit=args.limit, order_by=None, filter_expression=None, view=view)