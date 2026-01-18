from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1MigrateResourceRequestMigrateAutomlModelConfig(_messages.Message):
    """Config for migrating Model in automl.googleapis.com to Vertex AI's
  Model.

  Fields:
    model: Required. Full resource name of automl Model. Format:
      `projects/{project}/locations/{location}/models/{model}`.
    modelDisplayName: Optional. Display name of the model in Vertex AI. System
      will pick a display name if unspecified.
  """
    model = _messages.StringField(1)
    modelDisplayName = _messages.StringField(2)