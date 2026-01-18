from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.indexes import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
@base.ReleaseTracks(base.ReleaseTrack.GA)
class DescribeV1(base.DescribeCommand):
    """Gets detailed index information about the given index id.

  ## EXAMPLES

  Describe an index `123` of project `example` in region `us-central1`, run:

    $ {command} 123 --project=example --region=us-central1
  """

    @staticmethod
    def Args(parser):
        flags.AddIndexResourceArg(parser, 'to describe')

    def _Run(self, args, version):
        index_ref = args.CONCEPTS.index.Parse()
        region = index_ref.AsDict()['locationsId']
        with endpoint_util.AiplatformEndpointOverrides(version, region=region):
            return client.IndexesClient(version=version).Get(index_ref)

    def Run(self, args):
        return self._Run(args, constants.GA_VERSION)