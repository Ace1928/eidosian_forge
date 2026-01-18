from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsJobRunsTerminateRequest(_messages.Message):
    """A ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsJobRunsTe
  rminateRequest object.

  Fields:
    name: Required. Name of the `JobRun`. Format must be `projects/{project}/l
      ocations/{location}/deliveryPipelines/{deliveryPipeline}/releases/{relea
      se}/rollouts/{rollout}/jobRuns/{jobRun}`.
    terminateJobRunRequest: A TerminateJobRunRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    terminateJobRunRequest = _messages.MessageField('TerminateJobRunRequest', 2)