from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.builds import flags as build_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class DeleteAlpha(base.DeleteCommand):
    """Delete a Bitbucket Server config from Cloud Build."""

    @staticmethod
    def Args(parser):
        """Register flags for this command.

    Args:
      parser: An argparse.ArgumentParser-like object. It is mocked out in order
        to capture some information, but behaves like an ArgumentParser.
    """
        build_flags.AddRegionFlag(parser)
        parser.add_argument('CONFIG', help='The id of the Bitbucket Server Config')

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Nothing on success.
    """
        client = cloudbuild_util.GetClientInstance()
        messages = cloudbuild_util.GetMessagesModule()
        parent = properties.VALUES.core.project.Get(required=True)
        regionprop = properties.VALUES.builds.region.Get()
        bbs_region = args.region or regionprop or cloudbuild_util.DEFAULT_REGION
        config_id = args.CONFIG
        bbs_resource = resources.REGISTRY.Parse(None, collection='cloudbuild.projects.locations.bitbucketServerConfigs', api_version='v1', params={'projectsId': parent, 'locationsId': bbs_region, 'bitbucketServerConfigsId': config_id})
        deleted_op = client.projects_locations_bitbucketServerConfigs.Delete(messages.CloudbuildProjectsLocationsBitbucketServerConfigsDeleteRequest(name=bbs_resource.RelativeName()))
        op_resource = resources.REGISTRY.ParseRelativeName(deleted_op.name, collection='cloudbuild.projects.locations.operations')
        waiter.WaitFor(waiter.CloudOperationPollerNoResources(client.projects_locations_operations), op_resource, 'Deleting Bitbucket Server config')
        log.DeletedResource(bbs_resource)