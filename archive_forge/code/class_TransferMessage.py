from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferMessage(_messages.Message):
    """Represents a user facing message for a particular data transfer run.

  Enums:
    SeverityValueValuesEnum: Message severity.

  Fields:
    messageText: Message text.
    messageTime: Time when message was logged.
    severity: Message severity.
  """

    class SeverityValueValuesEnum(_messages.Enum):
        """Message severity.

    Values:
      MESSAGE_SEVERITY_UNSPECIFIED: No severity specified.
      INFO: Informational message.
      WARNING: Warning message.
      ERROR: Error message.
    """
        MESSAGE_SEVERITY_UNSPECIFIED = 0
        INFO = 1
        WARNING = 2
        ERROR = 3
    messageText = _messages.StringField(1)
    messageTime = _messages.StringField(2)
    severity = _messages.EnumField('SeverityValueValuesEnum', 3)