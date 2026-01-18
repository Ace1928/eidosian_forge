from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.custom_jobs import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import validation as common_validation
from googlecloudsdk.command_lib.ai.custom_jobs import custom_jobs_util
from googlecloudsdk.command_lib.ai.custom_jobs import flags
from googlecloudsdk.command_lib.ai.custom_jobs import validation
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class CreatePreGA(CreateGA):
    """Create a new custom job.

  This command will attempt to run the custom job immediately upon creation.

  ## EXAMPLES

  To create a job under project ``example'' in region
  ``us-central1'', run:

    $ {command} --region=us-central1 --project=example
    --worker-pool-spec=replica-count=1,machine-type='n1-highmem-2',container-image-uri='gcr.io/ucaip-test/ucaip-training-test'
    --display-name=test
  """
    _version = constants.BETA_VERSION