from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudResourcesettingsV1Value(_messages.Message):
    """The data in a setting value.

  Fields:
    booleanValue: Defines this value as being a boolean value.
    durationValue: Defines this value as being a Duration.
    enumValue: Defines this value as being a Enum.
    stringMapValue: Defines this value as being a StringMap.
    stringSetValue: Defines this value as being a StringSet.
    stringValue: Defines this value as being a string value.
  """
    booleanValue = _messages.BooleanField(1)
    durationValue = _messages.StringField(2)
    enumValue = _messages.MessageField('GoogleCloudResourcesettingsV1ValueEnumValue', 3)
    stringMapValue = _messages.MessageField('GoogleCloudResourcesettingsV1ValueStringMap', 4)
    stringSetValue = _messages.MessageField('GoogleCloudResourcesettingsV1ValueStringSet', 5)
    stringValue = _messages.StringField(6)