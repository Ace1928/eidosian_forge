from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IoCloudeventsV1CloudEventCloudEventAttributeValue(_messages.Message):
    """A IoCloudeventsV1CloudEventCloudEventAttributeValue object.

  Fields:
    ceBoolean: A boolean attribute.
    ceBytes: A byte attribute.
    ceInteger: A integer attribute.
    ceString: A string attribute.
    ceTimestamp: A string attribute.
    ceUri: A string attribute.
    ceUriRef: A string attribute.
  """
    ceBoolean = _messages.BooleanField(1)
    ceBytes = _messages.BytesField(2)
    ceInteger = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    ceString = _messages.StringField(4)
    ceTimestamp = _messages.StringField(5)
    ceUri = _messages.StringField(6)
    ceUriRef = _messages.StringField(7)