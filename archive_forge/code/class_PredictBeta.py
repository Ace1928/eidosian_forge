from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.endpoints import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import endpoints_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import region_util
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class PredictBeta(PredictGa):
    """Run Vertex AI online prediction.

     `{command}` sends a prediction request to Vertex AI endpoint for the
     given instances. This command will read up to 100 instances, though the
     service itself will accept instances up to the payload limit size
     (currently, 1.5MB).

  ## EXAMPLES

  To predict against an endpoint ``123'' under project ``example'' in region
  ``us-central1'', run:

    $ {command} 123 --project=example --region=us-central1
    --json-request=input.json
  """

    def Run(self, args):
        return _Run(args, constants.BETA_VERSION)