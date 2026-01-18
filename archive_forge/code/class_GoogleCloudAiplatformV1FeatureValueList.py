from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FeatureValueList(_messages.Message):
    """Container for list of values.

  Fields:
    values: A list of feature values. All of them should be the same data
      type.
  """
    values = _messages.MessageField('GoogleCloudAiplatformV1FeatureValue', 1, repeated=True)