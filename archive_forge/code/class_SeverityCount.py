from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SeverityCount(_messages.Message):
    """The number of occurrences created for a specific severity.

  Enums:
    SeverityValueValuesEnum: The severity of the occurrences.

  Fields:
    count: The number of occurrences with the severity.
    severity: The severity of the occurrences.
  """

    class SeverityValueValuesEnum(_messages.Enum):
        """The severity of the occurrences.

    Values:
      SEVERITY_UNSPECIFIED: Unknown Impact
      MINIMAL: Minimal Impact
      LOW: Low Impact
      MEDIUM: Medium Impact
      HIGH: High Impact
      CRITICAL: Critical Impact
    """
        SEVERITY_UNSPECIFIED = 0
        MINIMAL = 1
        LOW = 2
        MEDIUM = 3
        HIGH = 4
        CRITICAL = 5
    count = _messages.IntegerField(1)
    severity = _messages.EnumField('SeverityValueValuesEnum', 2)