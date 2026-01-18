from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamingQueryProgress(_messages.Message):
    """A StreamingQueryProgress object.

  Messages:
    DurationMillisValue: A DurationMillisValue object.
    EventTimeValue: A EventTimeValue object.
    ObservedMetricsValue: A ObservedMetricsValue object.

  Fields:
    batchDuration: A string attribute.
    batchId: A string attribute.
    durationMillis: A DurationMillisValue attribute.
    eventTime: A EventTimeValue attribute.
    name: A string attribute.
    observedMetrics: A ObservedMetricsValue attribute.
    runId: A string attribute.
    sink: A SinkProgress attribute.
    sources: A SourceProgress attribute.
    stateOperators: A StateOperatorProgress attribute.
    streamingQueryProgressId: A string attribute.
    timestamp: A string attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DurationMillisValue(_messages.Message):
        """A DurationMillisValue object.

    Messages:
      AdditionalProperty: An additional property for a DurationMillisValue
        object.

    Fields:
      additionalProperties: Additional properties of type DurationMillisValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DurationMillisValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EventTimeValue(_messages.Message):
        """A EventTimeValue object.

    Messages:
      AdditionalProperty: An additional property for a EventTimeValue object.

    Fields:
      additionalProperties: Additional properties of type EventTimeValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EventTimeValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ObservedMetricsValue(_messages.Message):
        """A ObservedMetricsValue object.

    Messages:
      AdditionalProperty: An additional property for a ObservedMetricsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ObservedMetricsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ObservedMetricsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    batchDuration = _messages.IntegerField(1)
    batchId = _messages.IntegerField(2)
    durationMillis = _messages.MessageField('DurationMillisValue', 3)
    eventTime = _messages.MessageField('EventTimeValue', 4)
    name = _messages.StringField(5)
    observedMetrics = _messages.MessageField('ObservedMetricsValue', 6)
    runId = _messages.StringField(7)
    sink = _messages.MessageField('SinkProgress', 8)
    sources = _messages.MessageField('SourceProgress', 9, repeated=True)
    stateOperators = _messages.MessageField('StateOperatorProgress', 10, repeated=True)
    streamingQueryProgressId = _messages.StringField(11)
    timestamp = _messages.StringField(12)