from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.declarative import flags as declarative_flags
from googlecloudsdk.command_lib.util.declarative.clients import declarative_client_base
from googlecloudsdk.command_lib.util.declarative.clients import kcc_client
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class ListResources(base.DeclarativeCommand):
    """List all resources supported by bulk-export."""
    detailed_help = _DETAILED_HELP

    @classmethod
    def Args(cls, parser):
        declarative_flags.AddListResourcesFlags(parser)
        parser.display_info.AddFormat(declarative_client_base.RESOURCE_LIST_FORMAT)

    def Run(self, args):
        client = kcc_client.KccClient()
        output = client.ListResources(project=args.project, organization=args.organization, folder=args.folder)
        return output