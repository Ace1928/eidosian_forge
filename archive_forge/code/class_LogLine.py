from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogLine(_messages.Message):
    """Application log line emitted while processing a request.

  Enums:
    SeverityValueValuesEnum: Severity of this log entry.

  Fields:
    logMessage: App-provided log message.
    severity: Severity of this log entry.
    sourceLocation: Where in the source code this log message was written.
    time: Approximate time when this log entry was made.
  """

    class SeverityValueValuesEnum(_messages.Enum):
        """Severity of this log entry.

    Values:
      DEFAULT: (0) The log entry has no assigned severity level.
      DEBUG: (100) Debug or trace information.
      INFO: (200) Routine information, such as ongoing status or performance.
      NOTICE: (300) Normal but significant events, such as start up, shut
        down, or a configuration change.
      WARNING: (400) Warning events might cause problems.
      ERROR: (500) Error events are likely to cause problems.
      CRITICAL: (600) Critical events cause more severe problems or outages.
      ALERT: (700) A person must take an action immediately.
      EMERGENCY: (800) One or more systems are unusable.
    """
        DEFAULT = 0
        DEBUG = 1
        INFO = 2
        NOTICE = 3
        WARNING = 4
        ERROR = 5
        CRITICAL = 6
        ALERT = 7
        EMERGENCY = 8
    logMessage = _messages.StringField(1)
    severity = _messages.EnumField('SeverityValueValuesEnum', 2)
    sourceLocation = _messages.MessageField('SourceLocation', 3)
    time = _messages.StringField(4)