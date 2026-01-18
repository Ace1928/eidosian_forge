from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.apphub import service_projects as apis
from googlecloudsdk.api_lib.apphub import utils as api_lib_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.apphub import flags
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class RemoveAlpha(base.DeleteCommand):
    """Remove an Apphub service project."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        flags.AddRemoveServiceProjectFlags(parser)

    def Run(self, args):
        """Run the remove command."""
        client = apis.ServiceProjectsClient(release_track=base.ReleaseTrack.ALPHA)
        service_project_ref = api_lib_utils.GetServiceProjectRef(args)
        return client.Remove(service_project=service_project_ref.RelativeName(), async_flag=args.async_)