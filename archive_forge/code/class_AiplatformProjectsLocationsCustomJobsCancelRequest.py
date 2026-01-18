from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsCustomJobsCancelRequest(_messages.Message):
    """A AiplatformProjectsLocationsCustomJobsCancelRequest object.

  Fields:
    googleCloudAiplatformV1CancelCustomJobRequest: A
      GoogleCloudAiplatformV1CancelCustomJobRequest resource to be passed as
      the request body.
    name: Required. The name of the CustomJob to cancel. Format:
      `projects/{project}/locations/{location}/customJobs/{custom_job}`
  """
    googleCloudAiplatformV1CancelCustomJobRequest = _messages.MessageField('GoogleCloudAiplatformV1CancelCustomJobRequest', 1)
    name = _messages.StringField(2, required=True)