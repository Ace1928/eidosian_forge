from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeaturestoresEntityTypesCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeaturestoresEntityTypesCreateRequest
  object.

  Fields:
    entityTypeId: Required. The ID to use for the EntityType, which will
      become the final component of the EntityType's resource name. This value
      may be up to 60 characters, and valid characters are `[a-z0-9_]`. The
      first character cannot be a number. The value must be unique within a
      featurestore.
    googleCloudAiplatformV1EntityType: A GoogleCloudAiplatformV1EntityType
      resource to be passed as the request body.
    parent: Required. The resource name of the Featurestore to create
      EntityTypes. Format:
      `projects/{project}/locations/{location}/featurestores/{featurestore}`
  """
    entityTypeId = _messages.StringField(1)
    googleCloudAiplatformV1EntityType = _messages.MessageField('GoogleCloudAiplatformV1EntityType', 2)
    parent = _messages.StringField(3, required=True)