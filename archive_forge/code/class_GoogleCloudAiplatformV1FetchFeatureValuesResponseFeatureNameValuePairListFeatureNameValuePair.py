from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FetchFeatureValuesResponseFeatureNameValuePairListFeatureNameValuePair(_messages.Message):
    """Feature name & value pair.

  Fields:
    name: Feature short name.
    value: Feature value.
  """
    name = _messages.StringField(1)
    value = _messages.MessageField('GoogleCloudAiplatformV1FeatureValue', 2)