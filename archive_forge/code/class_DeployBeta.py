from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.api_lib.app import runtime_builders
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import deploy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class DeployBeta(base.SilentCommand):
    """Deploy the local code and/or configuration of your app to App Engine."""

    @staticmethod
    def Args(parser):
        """Get arguments for this command."""
        deploy_util.ArgsDeploy(parser)

    def Run(self, args):
        runtime_builder_strategy = deploy_util.GetRuntimeBuilderStrategy(base.ReleaseTrack.BETA)
        api_client = appengine_api_client.GetApiClientForTrack(self.ReleaseTrack())
        return deploy_util.RunDeploy(args, api_client, use_beta_stager=True, runtime_builder_strategy=runtime_builder_strategy, parallel_build=True, flex_image_build_option=deploy_util.GetFlexImageBuildOption(default_strategy=deploy_util.FlexImageBuildOptions.ON_SERVER))