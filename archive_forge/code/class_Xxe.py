from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class Xxe(_messages.Message):
    """Information reported for an XXE.

  Enums:
    PayloadLocationValueValuesEnum: Location within the request where the
      payload was placed.

  Fields:
    payloadLocation: Location within the request where the payload was placed.
    payloadValue: The XML string that triggered the XXE vulnerability. Non-
      payload values might be redacted.
  """

    class PayloadLocationValueValuesEnum(_messages.Enum):
        """Location within the request where the payload was placed.

    Values:
      LOCATION_UNSPECIFIED: Unknown Location.
      COMPLETE_REQUEST_BODY: The XML payload replaced the complete request
        body.
    """
        LOCATION_UNSPECIFIED = 0
        COMPLETE_REQUEST_BODY = 1
    payloadLocation = _messages.EnumField('PayloadLocationValueValuesEnum', 1)
    payloadValue = _messages.StringField(2)