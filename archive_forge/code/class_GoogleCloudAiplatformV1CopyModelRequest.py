from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1CopyModelRequest(_messages.Message):
    """Request message for ModelService.CopyModel.

  Fields:
    encryptionSpec: Customer-managed encryption key options. If this is set,
      then the Model copy will be encrypted with the provided encryption key.
    modelId: Optional. Copy source_model into a new Model with this ID. The ID
      will become the final component of the model resource name. This value
      may be up to 63 characters, and valid characters are `[a-z0-9_-]`. The
      first character cannot be a number or hyphen.
    parentModel: Optional. Specify this field to copy source_model into this
      existing Model as a new version. Format:
      `projects/{project}/locations/{location}/models/{model}`
    sourceModel: Required. The resource name of the Model to copy. That Model
      must be in the same Project. Format:
      `projects/{project}/locations/{location}/models/{model}`
  """
    encryptionSpec = _messages.MessageField('GoogleCloudAiplatformV1EncryptionSpec', 1)
    modelId = _messages.StringField(2)
    parentModel = _messages.StringField(3)
    sourceModel = _messages.StringField(4)