from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import xpn_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.xpn import flags
class GetHostProject(base.Command):
    """Get the shared VPC host project that the given project is associated with.
  """
    detailed_help = {'EXAMPLES': '\n          If `service-project1` and `service-project2` are associated service\n          projects of the shared VPC host project `host-project`,\n\n            $ {command} service-project1\n\n          will show the `host-project` project.\n      '}

    @staticmethod
    def Args(parser):
        flags.GetProjectIdArgument('get the host project for').AddToParser(parser)

    def Run(self, args):
        xpn_client = xpn_api.GetXpnClient(self.ReleaseTrack())
        return xpn_client.GetHostProject(args.project)