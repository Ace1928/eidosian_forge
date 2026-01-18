from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsRetryJobRequest(_messages.Message):
    """A
  ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsRetryJobRequest
  object.

  Fields:
    retryJobRequest: A RetryJobRequest resource to be passed as the request
      body.
    rollout: Required. Name of the Rollout. Format is `projects/{project}/loca
      tions/{location}/deliveryPipelines/{deliveryPipeline}/releases/{release}
      /rollouts/{rollout}`.
  """
    retryJobRequest = _messages.MessageField('RetryJobRequest', 1)
    rollout = _messages.StringField(2, required=True)