from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudFunctionsV2betaStateMessage(_messages.Message):
    """Informational messages about the state of the Cloud Function or
  Operation.

  Enums:
    SeverityValueValuesEnum: Severity of the state message.

  Fields:
    message: The message.
    severity: Severity of the state message.
    type: One-word CamelCase type of the state message.
  """

    class SeverityValueValuesEnum(_messages.Enum):
        """Severity of the state message.

    Values:
      SEVERITY_UNSPECIFIED: Not specified. Invalid severity.
      ERROR: ERROR-level severity.
      WARNING: WARNING-level severity.
      INFO: INFO-level severity.
    """
        SEVERITY_UNSPECIFIED = 0
        ERROR = 1
        WARNING = 2
        INFO = 3
    message = _messages.StringField(1)
    severity = _messages.EnumField('SeverityValueValuesEnum', 2)
    type = _messages.StringField(3)