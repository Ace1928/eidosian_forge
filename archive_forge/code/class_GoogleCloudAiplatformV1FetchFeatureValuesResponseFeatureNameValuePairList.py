from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FetchFeatureValuesResponseFeatureNameValuePairList(_messages.Message):
    """Response structure in the format of key (feature name) and (feature)
  value pair.

  Fields:
    features: List of feature names and values.
  """
    features = _messages.MessageField('GoogleCloudAiplatformV1FetchFeatureValuesResponseFeatureNameValuePairListFeatureNameValuePair', 1, repeated=True)