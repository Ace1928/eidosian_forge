from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai import operations
from googlecloudsdk.api_lib.ai.index_endpoints import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import index_endpoints_util
from googlecloudsdk.command_lib.ai import operations_util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class UndeployIndexV1Beta1(UndeployIndexV1):
    """Undeploy an index from a Vertex AI index endpoint.

  ## EXAMPLES

  To undeploy the deployed-index ``deployed-index-345'' from an index endpoint
  ``456'' under project ``example'' in region ``us-central1'', run:

    $ {command} 456 --project=example --region=us-central1
    --deployed-index-id=deployed-index-345
  """

    def Run(self, args):
        return self._Run(args, constants.BETA_VERSION)