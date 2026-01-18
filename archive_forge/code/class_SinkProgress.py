from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SinkProgress(_messages.Message):
    """A SinkProgress object.

  Messages:
    MetricsValue: A MetricsValue object.

  Fields:
    description: A string attribute.
    metrics: A MetricsValue attribute.
    numOutputRows: A string attribute.
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
    metrics = _messages.MessageField('MetricsValue', 2)
    numOutputRows = _messages.IntegerField(3)