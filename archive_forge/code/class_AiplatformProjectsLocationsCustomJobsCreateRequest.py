from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsCustomJobsCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsCustomJobsCreateRequest object.

  Fields:
    googleCloudAiplatformV1CustomJob: A GoogleCloudAiplatformV1CustomJob
      resource to be passed as the request body.
    parent: Required. The resource name of the Location to create the
      CustomJob in. Format: `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1CustomJob = _messages.MessageField('GoogleCloudAiplatformV1CustomJob', 1)
    parent = _messages.StringField(2, required=True)