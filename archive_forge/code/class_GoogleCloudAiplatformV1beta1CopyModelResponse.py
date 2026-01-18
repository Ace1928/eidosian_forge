from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1CopyModelResponse(_messages.Message):
    """Response message of ModelService.CopyModel operation.

  Fields:
    model: The name of the copied Model resource. Format:
      `projects/{project}/locations/{location}/models/{model}`
    modelVersionId: Output only. The version ID of the model that is copied.
  """
    model = _messages.StringField(1)
    modelVersionId = _messages.StringField(2)