from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.networking.interconnects import interconnects
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.edge_cloud.networking import resource_args
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class GetStatus(base.Command):
    """Get the diagnostics of a specified Distributed Cloud Edge Network interconnect.

  *{command}* is used to get the diagnostics of a Distributed Cloud Edge Network
  interconnect.
  """
    detailed_help = {'DESCRIPTION': DESCRIPTION, 'EXAMPLES': EXAMPLES}

    @staticmethod
    def Args(parser):
        resource_args.AddInterconnectResourceArg(parser, 'to get diagnostics', True)

    def Run(self, args):
        interconnects_client = interconnects.InterconnectsClient(self.ReleaseTrack())
        interconnect_ref = args.CONCEPTS.interconnect.Parse()
        return interconnects_client.GetStatus(interconnect_ref)