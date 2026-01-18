from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TimestampedEvent(_messages.Message):
    """An event that occured in the operation assigned to the worker and the
  time of occurance.

  Messages:
    DataValue: The event data.

  Fields:
    data: The event data.
    timestamp: The time when the event happened.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DataValue(_messages.Message):
        """The event data.

    Messages:
      AdditionalProperty: An additional property for a DataValue object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DataValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    data = _messages.MessageField('DataValue', 1)
    timestamp = _messages.StringField(2)