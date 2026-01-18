from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FeatureViewDataKey(_messages.Message):
    """Lookup key for a feature view.

  Fields:
    compositeKey: The actual Entity ID will be composed from this struct. This
      should match with the way ID is defined in the FeatureView spec.
    key: String key to use for lookup.
  """
    compositeKey = _messages.MessageField('GoogleCloudAiplatformV1FeatureViewDataKeyCompositeKey', 1)
    key = _messages.StringField(2)