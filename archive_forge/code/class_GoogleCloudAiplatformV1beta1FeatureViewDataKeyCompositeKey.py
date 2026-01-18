from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FeatureViewDataKeyCompositeKey(_messages.Message):
    """ID that is comprised from several parts (columns).

  Fields:
    parts: Parts to construct Entity ID. Should match with the same ID columns
      as defined in FeatureView in the same order.
  """
    parts = _messages.StringField(1, repeated=True)