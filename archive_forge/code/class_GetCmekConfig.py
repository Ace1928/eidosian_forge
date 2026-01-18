from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.tasks import GetApiAdapter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import flags
from googlecloudsdk.command_lib.tasks import parsers
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class GetCmekConfig(base.Command):
    """Get CMEK configuration for Cloud Tasks in the specified location."""
    detailed_help = {'DESCRIPTION': '          {description}\n          ', 'EXAMPLES': '          To get a CMEK config:\n\n              $ {command} --location=my-location\n         '}

    @staticmethod
    def Args(parser):
        flags.DescribeCmekConfigResourceFlag(parser)

    def Run(self, args):
        api = GetApiAdapter(self.ReleaseTrack())
        cmek_config_client = api.cmek_config
        project_id, location_id = parsers.ParseKmsDescribeArgs(args)
        cmek_config = cmek_config_client.GetCmekConfig(project_id, location_id)
        return cmek_config