from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsIndexesCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsIndexesCreateRequest object.

  Fields:
    googleCloudAiplatformV1Index: A GoogleCloudAiplatformV1Index resource to
      be passed as the request body.
    parent: Required. The resource name of the Location to create the Index
      in. Format: `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1Index = _messages.MessageField('GoogleCloudAiplatformV1Index', 1)
    parent = _messages.StringField(2, required=True)