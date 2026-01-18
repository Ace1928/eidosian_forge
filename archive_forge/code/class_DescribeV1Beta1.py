from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.deployment_resource_pools import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import region_util
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class DescribeV1Beta1(base.DescribeCommand):
    """Describe a Vertex AI deployment resource pool.

  This command describes a deployment resource pool with a provided deployment
  resource pool.

  ## EXAMPLES

  To describe a deployment resource pool with name ''example'' in region
  ''us-central1'', run:

    $ {command} example --region=us-central1
  """

    @staticmethod
    def Args(parser):
        return _ArgsBeta(parser)

    def Run(self, args):
        return _RunBeta(args)