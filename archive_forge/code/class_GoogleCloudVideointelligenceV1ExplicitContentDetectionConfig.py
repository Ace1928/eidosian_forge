from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1ExplicitContentDetectionConfig(_messages.Message):
    """Config for EXPLICIT_CONTENT_DETECTION.

  Fields:
    model: Model to use for explicit content detection. Supported values:
      "builtin/stable" (the default if unset) and "builtin/latest".
  """
    model = _messages.StringField(1)