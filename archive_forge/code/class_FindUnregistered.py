from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.apphub import discovered_services as apis
from googlecloudsdk.api_lib.apphub import utils as api_lib_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.apphub import flags
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class FindUnregistered(base.ListCommand):
    """List unregistered Apphub discovered services."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        flags.AddFindUnregisteredServiceFlags(parser)
        parser.display_info.AddFormat(_FORMAT)
        parser.display_info.AddUriFunc(api_lib_utils.MakeGetUriFunc('apphub.projects.locations.discoveredServices'))

    def Run(self, args):
        """Run the find unregistered service command."""
        client = apis.DiscoveredServicesClient()
        location_ref = args.CONCEPTS.location.Parse()
        return client.FindUnregistered(limit=args.limit, page_size=args.page_size, parent=location_ref.RelativeName())