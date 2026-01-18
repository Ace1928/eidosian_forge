from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceProgress(_messages.Message):
    """A SourceProgress object.

  Messages:
    MetricsValue: A MetricsValue object.

  Fields:
    description: A string attribute.
    endOffset: A string attribute.
    inputRowsPerSecond: A number attribute.
    latestOffset: A string attribute.
    metrics: A MetricsValue attribute.
    numInputRows: A string attribute.
    processedRowsPerSecond: A number attribute.
    startOffset: A string attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetricsValue(_messages.Message):
        """A MetricsValue object.

    Messages:
      AdditionalProperty: An additional property for a MetricsValue object.

    Fields:
      additionalProperties: Additional properties of type MetricsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetricsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    description = _messages.StringField(1)
    endOffset = _messages.StringField(2)
    inputRowsPerSecond = _messages.FloatField(3)
    latestOffset = _messages.StringField(4)
    metrics = _messages.MessageField('MetricsValue', 5)
    numInputRows = _messages.IntegerField(6)
    processedRowsPerSecond = _messages.FloatField(7)
    startOffset = _messages.StringField(8)