from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsIgnoreJobRequest(_messages.Message):
    """A ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsIgnoreJob
  Request object.

  Fields:
    ignoreJobRequest: A IgnoreJobRequest resource to be passed as the request
      body.
    rollout: Required. Name of the Rollout. Format is `projects/{project}/loca
      tions/{location}/deliveryPipelines/{deliveryPipeline}/releases/{release}
      /rollouts/{rollout}`.
  """
    ignoreJobRequest = _messages.MessageField('IgnoreJobRequest', 1)
    rollout = _messages.StringField(2, required=True)