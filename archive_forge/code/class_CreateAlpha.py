from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.cloudbuild import bitbucketserver_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class CreateAlpha(base.CreateCommand):
    """Update a Bitbucket Server config for use by Cloud Build."""

    @staticmethod
    def Args(parser):
        """Register flags for this command.

    Args:
      parser: An argparse.ArgumentParser-like object. It is mocked out in order
        to capture some information, but behaves like an ArgumentParser.
    """
        parser = bitbucketserver_flags.AddBitbucketServerConfigUpdateArgs(parser)
        parser.display_info.AddFormat("\n          table(\n            name,\n            createTime.date('%Y-%m-%dT%H:%M:%S%Oz', undefined='-'),\n            host_uri\n          )\n        ")

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Some value that we want to have printed later.
    """
        client = cloudbuild_util.GetClientInstance()
        messages = cloudbuild_util.GetMessagesModule()
        config_id = args.CONFIG
        bbs = cloudbuild_util.BitbucketServerConfigFromArgs(args, True)
        parent = properties.VALUES.core.project.Get(required=True)
        regionprop = properties.VALUES.builds.region.Get()
        bbs_region = args.region or regionprop or cloudbuild_util.DEFAULT_REGION
        bbs_resource = resources.REGISTRY.Parse(None, collection='cloudbuild.projects.locations.bitbucketServerConfigs', api_version='v1', params={'projectsId': parent, 'locationsId': bbs_region, 'bitbucketServerConfigsId': config_id})
        update_mask = cloudbuild_util.MessageToFieldPaths(bbs)
        req = messages.CloudbuildProjectsLocationsBitbucketServerConfigsPatchRequest(name=bbs_resource.RelativeName(), bitbucketServerConfig=bbs, updateMask=','.join(update_mask))
        updated_op = client.projects_locations_bitbucketServerConfigs.Patch(req)
        op_resource = resources.REGISTRY.ParseRelativeName(updated_op.name, collection='cloudbuild.projects.locations.operations')
        updated_bbs = waiter.WaitFor(waiter.CloudOperationPoller(client.projects_locations_bitbucketServerConfigs, client.projects_locations_operations), op_resource, 'Updating Bitbucket Server config')
        log.UpdatedResource(bbs_resource)
        return updated_bbs