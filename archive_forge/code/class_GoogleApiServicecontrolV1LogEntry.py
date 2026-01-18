from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServicecontrolV1LogEntry(_messages.Message):
    """An individual log entry.

  Enums:
    SeverityValueValuesEnum: The severity of the log entry. The default value
      is `LogSeverity.DEFAULT`.

  Messages:
    LabelsValue: A set of user-defined (key, value) data that provides
      additional information about the log entry.
    ProtoPayloadValue: The log entry payload, represented as a protocol buffer
      that is expressed as a JSON object. The only accepted type currently is
      AuditLog.
    StructPayloadValue: The log entry payload, represented as a structure that
      is expressed as a JSON object.

  Fields:
    httpRequest: Optional. Information about the HTTP request associated with
      this log entry, if applicable.
    insertId: A unique ID for the log entry used for deduplication. If
      omitted, the implementation will generate one based on operation_id.
    labels: A set of user-defined (key, value) data that provides additional
      information about the log entry.
    name: Required. The log to which this log entry belongs. Examples:
      `"syslog"`, `"book_log"`.
    operation: Optional. Information about an operation associated with the
      log entry, if applicable.
    protoPayload: The log entry payload, represented as a protocol buffer that
      is expressed as a JSON object. The only accepted type currently is
      AuditLog.
    severity: The severity of the log entry. The default value is
      `LogSeverity.DEFAULT`.
    sourceLocation: Optional. Source code location information associated with
      the log entry, if any.
    structPayload: The log entry payload, represented as a structure that is
      expressed as a JSON object.
    textPayload: The log entry payload, represented as a Unicode string
      (UTF-8).
    timestamp: The time the event described by the log entry occurred. If
      omitted, defaults to operation start time.
    trace: Optional. Resource name of the trace associated with the log entry,
      if any. If this field contains a relative resource name, you can assume
      the name is relative to `//tracing.googleapis.com`. Example:
      `projects/my-projectid/traces/06796866738c859f2f19b7cfb3214824`
  """

    class SeverityValueValuesEnum(_messages.Enum):
        """The severity of the log entry. The default value is
    `LogSeverity.DEFAULT`.

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """A set of user-defined (key, value) data that provides additional
    information about the log entry.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ProtoPayloadValue(_messages.Message):
        """The log entry payload, represented as a protocol buffer that is
    expressed as a JSON object. The only accepted type currently is AuditLog.

    Messages:
      AdditionalProperty: An additional property for a ProtoPayloadValue
        object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ProtoPayloadValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class StructPayloadValue(_messages.Message):
        """The log entry payload, represented as a structure that is expressed as
    a JSON object.

    Messages:
      AdditionalProperty: An additional property for a StructPayloadValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a StructPayloadValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    httpRequest = _messages.MessageField('GoogleApiServicecontrolV1HttpRequest', 1)
    insertId = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    name = _messages.StringField(4)
    operation = _messages.MessageField('GoogleApiServicecontrolV1LogEntryOperation', 5)
    protoPayload = _messages.MessageField('ProtoPayloadValue', 6)
    severity = _messages.EnumField('SeverityValueValuesEnum', 7)
    sourceLocation = _messages.MessageField('GoogleApiServicecontrolV1LogEntrySourceLocation', 8)
    structPayload = _messages.MessageField('StructPayloadValue', 9)
    textPayload = _messages.StringField(10)
    timestamp = _messages.StringField(11)
    trace = _messages.StringField(12)