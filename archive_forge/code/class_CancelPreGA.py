from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.custom_jobs import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai.custom_jobs import flags
from googlecloudsdk.command_lib.ai.custom_jobs import validation
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class CancelPreGA(CancelGA):
    """Cancel a running custom job.

  If the job is already finished,
  the command will not perform any operation.

  To cancel a job ``123'' under project ``example'' in region
  ``us-central1'', run:

    $ {command} 123 --project=example --region=us-central1
  """
    _api_version = constants.BETA_VERSION

    def _CommandPrefix(self):
        return 'gcloud ' + self.ReleaseTrack().prefix