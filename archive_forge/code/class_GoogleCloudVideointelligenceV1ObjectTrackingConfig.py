from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1ObjectTrackingConfig(_messages.Message):
    """Config for OBJECT_TRACKING.

  Fields:
    model: Model to use for object tracking. Supported values:
      "builtin/stable" (the default if unset) and "builtin/latest".
  """
    model = _messages.StringField(1)