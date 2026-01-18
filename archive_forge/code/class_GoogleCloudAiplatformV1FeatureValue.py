from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FeatureValue(_messages.Message):
    """Value for a feature.

  Fields:
    boolArrayValue: A list of bool type feature value.
    boolValue: Bool type feature value.
    bytesValue: Bytes feature value.
    doubleArrayValue: A list of double type feature value.
    doubleValue: Double type feature value.
    int64ArrayValue: A list of int64 type feature value.
    int64Value: Int64 feature value.
    metadata: Metadata of feature value.
    stringArrayValue: A list of string type feature value.
    stringValue: String feature value.
  """
    boolArrayValue = _messages.MessageField('GoogleCloudAiplatformV1BoolArray', 1)
    boolValue = _messages.BooleanField(2)
    bytesValue = _messages.BytesField(3)
    doubleArrayValue = _messages.MessageField('GoogleCloudAiplatformV1DoubleArray', 4)
    doubleValue = _messages.FloatField(5)
    int64ArrayValue = _messages.MessageField('GoogleCloudAiplatformV1Int64Array', 6)
    int64Value = _messages.IntegerField(7)
    metadata = _messages.MessageField('GoogleCloudAiplatformV1FeatureValueMetadata', 8)
    stringArrayValue = _messages.MessageField('GoogleCloudAiplatformV1StringArray', 9)
    stringValue = _messages.StringField(10)