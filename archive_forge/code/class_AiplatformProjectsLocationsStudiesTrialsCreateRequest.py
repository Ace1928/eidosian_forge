from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsStudiesTrialsCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsStudiesTrialsCreateRequest object.

  Fields:
    googleCloudAiplatformV1Trial: A GoogleCloudAiplatformV1Trial resource to
      be passed as the request body.
    parent: Required. The resource name of the Study to create the Trial in.
      Format: `projects/{project}/locations/{location}/studies/{study}`
  """
    googleCloudAiplatformV1Trial = _messages.MessageField('GoogleCloudAiplatformV1Trial', 1)
    parent = _messages.StringField(2, required=True)