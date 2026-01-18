from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ModelOriginalModelInfo(_messages.Message):
    """Contains information about the original Model if this Model is a copy.

  Fields:
    model: Output only. The resource name of the Model this Model is a copy
      of, including the revision. Format:
      `projects/{project}/locations/{location}/models/{model_id}@{version_id}`
  """
    model = _messages.StringField(1)