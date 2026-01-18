from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.indexes import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import region_util
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ListV1(base.ListCommand):
    """Lists the indexes of the given project and region.

  ## EXAMPLES

  Lists the indexes of project `example` in region `us-central1`, run:

    $ {command} --project=example --region=us-central1
  """

    @staticmethod
    def Args(parser):
        flags.AddRegionResourceArg(parser, 'to list indexes', prompt_func=region_util.GetPromptForRegionFunc(constants.SUPPORTED_OP_REGIONS))

    def _Run(self, args, version):
        region_ref = args.CONCEPTS.region.Parse()
        region = region_ref.AsDict()['locationsId']
        with endpoint_util.AiplatformEndpointOverrides(version, region=region):
            return client.IndexesClient(version=version).List(region_ref=region_ref)

    def Run(self, args):
        return self._Run(args, constants.GA_VERSION)