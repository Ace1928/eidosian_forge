from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1ShotChangeDetectionConfig(_messages.Message):
    """Config for SHOT_CHANGE_DETECTION.

  Fields:
    model: Model to use for shot change detection. Supported values:
      "builtin/stable" (the default if unset), "builtin/latest", and
      "builtin/legacy".
  """
    model = _messages.StringField(1)