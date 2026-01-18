from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDatasetsCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsDatasetsCreateRequest object.

  Fields:
    googleCloudAiplatformV1Dataset: A GoogleCloudAiplatformV1Dataset resource
      to be passed as the request body.
    parent: Required. The resource name of the Location to create the Dataset
      in. Format: `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1Dataset = _messages.MessageField('GoogleCloudAiplatformV1Dataset', 1)
    parent = _messages.StringField(2, required=True)