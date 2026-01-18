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
@base.ReleaseTracks(base.ReleaseTrack.GA)
class CancelGA(base.SilentCommand):
    """Cancel a running custom job.

  If the job is already finished,
  the command will not perform any operation.

  ## EXAMPLES

  To cancel a job ``123'' under project ``example'' in region
  ``us-central1'', run:

    $ {command} 123 --project=example --region=us-central1
  """
    _api_version = constants.GA_VERSION

    @staticmethod
    def Args(parser):
        flags.AddCustomJobResourceArg(parser, 'to cancel')

    def _CommandPrefix(self):
        return 'gcloud'

    def Run(self, args):
        custom_job_ref = args.CONCEPTS.custom_job.Parse()
        region = custom_job_ref.AsDict()['locationsId']
        validation.ValidateRegion(region)
        with endpoint_util.AiplatformEndpointOverrides(version=self._api_version, region=region):
            job_name = custom_job_ref.RelativeName()
            response = client.CustomJobsClient(version=self._api_version).Cancel(job_name)
            log.status.Print(_CUSTOM_JOB_CANCEL_DISPLAY_MESSAGE.format(job_name=job_name, command_prefix=self._CommandPrefix()))
            return response