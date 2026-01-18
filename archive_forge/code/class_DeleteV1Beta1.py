from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai import operations
from googlecloudsdk.api_lib.ai.deployment_resource_pools import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import deployment_resource_pools_util
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import operations_util
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.core.console import console_io
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class DeleteV1Beta1(base.DeleteCommand):
    """Delete an existing Vertex AI deployment resource pool.

  ## EXAMPLES

  To delete a deployment resource pool ``123'' under project ``example'' in
  region ``us-central1'', run:

    $ {command} 123 --project=example --region=us-central1
  """

    @staticmethod
    def Args(parser):
        return _ArgsBeta(parser)

    def Run(self, args):
        return _RunBeta(args)