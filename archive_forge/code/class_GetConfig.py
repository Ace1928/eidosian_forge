from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.interconnects import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.interconnects import flags
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class GetConfig(base.DescribeCommand):
    """Get MACsec configuration of a Compute Engine interconnect.

    *{command}* displays all MACsec configuration data associated with Compute
    Engine interconnect in a project.
  """
    INTERCONNECT_ARG = None

    @classmethod
    def Args(cls, parser):
        cls.INTERCONNECT_ARG = flags.InterconnectArgument()
        cls.INTERCONNECT_ARG.AddArgument(parser, operation_type='describe')

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        ref = self.INTERCONNECT_ARG.ResolveAsResource(args, holder.resources)
        interconnect = client.Interconnect(ref, compute_client=holder.client)
        return interconnect.GetMacsecConfig()