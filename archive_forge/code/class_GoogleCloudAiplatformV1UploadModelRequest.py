from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1UploadModelRequest(_messages.Message):
    """Request message for ModelService.UploadModel.

  Fields:
    model: Required. The Model to create.
    modelId: Optional. The ID to use for the uploaded Model, which will become
      the final component of the model resource name. This value may be up to
      63 characters, and valid characters are `[a-z0-9_-]`. The first
      character cannot be a number or hyphen.
    parentModel: Optional. The resource name of the model into which to upload
      the version. Only specify this field when uploading a new version.
    serviceAccount: Optional. The user-provided custom service account to use
      to do the model upload. If empty, [Vertex AI Service
      Agent](https://cloud.google.com/vertex-ai/docs/general/access-
      control#service-agents) will be used to access resources needed to
      upload the model. This account must belong to the target project where
      the model is uploaded to, i.e., the project specified in the `parent`
      field of this request and have necessary read permissions (to Google
      Cloud Storage, Artifact Registry, etc.).
  """
    model = _messages.MessageField('GoogleCloudAiplatformV1Model', 1)
    modelId = _messages.StringField(2)
    parentModel = _messages.StringField(3)
    serviceAccount = _messages.StringField(4)