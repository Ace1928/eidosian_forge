from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import xpn_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.xpn import flags
from googlecloudsdk.command_lib.compute.xpn import util as command_lib_util
class ListAssociatedResources(base.ListCommand):
    """List the resources associated with the given shared VPC host project.
  """
    detailed_help = {'EXAMPLES': '\n          If `service-project1` and `service-project2` are associated service\n          projects of the shared VPC host project `host-project`,\n\n            $ {command} host-project\n\n          yields the output\n\n            RESOURCE_ID         RESOURCE_TYPE\n            service-project1    PROJECT\n            service-project2    PROJECT\n'}

    @staticmethod
    def Args(parser):
        flags.GetProjectIdArgument('get associated resources for').AddToParser(parser)
        parser.display_info.AddFormat(command_lib_util.XPN_RESOURCE_ID_FORMAT)

    def Run(self, args):
        xpn_client = xpn_api.GetXpnClient(self.ReleaseTrack())
        return xpn_client.ListEnabledResources(args.project)