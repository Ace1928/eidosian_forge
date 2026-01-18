from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecordSummary(_messages.Message):
    """LINT.IfChange(RecordSummary) Summary of the underlying Record.

  Enums:
    StatusValueValuesEnum: Output only. Status of the underlying Run of this
      Record

  Messages:
    RecordDataValue: Output only. Key-value pairs representing underlying
      record data, e.g. "status", "SUCCESS"

  Fields:
    createTime: Output only. The time the Record was created.
    record: Output only. Summarized record.
    recordData: Output only. Key-value pairs representing underlying record
      data, e.g. "status", "SUCCESS"
    status: Output only. Status of the underlying Run of this Record
    type: Output only. Identifier of underlying data. e.g.
      `cloudbuild.googleapis.com/PipelineRun`
    updateTime: Output only. The time the Record was updated.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """Output only. Status of the underlying Run of this Record

    Values:
      STATUS_UNSPECIFIED: Default enum type; should not be used.
      SUCCESS: Run was successful
      FAILURE: Run failed
      TIMEOUT: Run timed out
      CANCELLED: Run got cancelled
      IN_PROGRESS: Run is in progress
      QUEUED: Run is queued
    """
        STATUS_UNSPECIFIED = 0
        SUCCESS = 1
        FAILURE = 2
        TIMEOUT = 3
        CANCELLED = 4
        IN_PROGRESS = 5
        QUEUED = 6

    @encoding.MapUnrecognizedFields('additionalProperties')
    class RecordDataValue(_messages.Message):
        """Output only. Key-value pairs representing underlying record data, e.g.
    "status", "SUCCESS"

    Messages:
      AdditionalProperty: An additional property for a RecordDataValue object.

    Fields:
      additionalProperties: Additional properties of type RecordDataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a RecordDataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    record = _messages.StringField(2)
    recordData = _messages.MessageField('RecordDataValue', 3)
    status = _messages.EnumField('StatusValueValuesEnum', 4)
    type = _messages.StringField(5)
    updateTime = _messages.StringField(6)