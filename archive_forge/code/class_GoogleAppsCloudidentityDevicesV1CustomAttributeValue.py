from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsCloudidentityDevicesV1CustomAttributeValue(_messages.Message):
    """Additional custom attribute values may be one of these types

  Fields:
    boolValue: Represents a boolean value.
    numberValue: Represents a double value.
    stringValue: Represents a string value.
  """
    boolValue = _messages.BooleanField(1)
    numberValue = _messages.FloatField(2)
    stringValue = _messages.StringField(3)