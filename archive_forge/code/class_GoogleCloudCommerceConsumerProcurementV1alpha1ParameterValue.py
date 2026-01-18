from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1ParameterValue(_messages.Message):
    """A GoogleCloudCommerceConsumerProcurementV1alpha1ParameterValue object.

  Fields:
    doubleValue: Represents a double value.
    int64Value: Represents an int64 value.
    stringValue: Represents a string value.
  """
    doubleValue = _messages.FloatField(1)
    int64Value = _messages.IntegerField(2)
    stringValue = _messages.StringField(3)