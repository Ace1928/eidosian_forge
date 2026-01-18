from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class PercentagesValue(_messages.Message):
    """Maps service configuration IDs to their corresponding traffic
    percentage. Key is the service configuration ID, Value is the traffic
    percentage which must be greater than 0.0 and the sum must equal to 100.0.

    Messages:
      AdditionalProperty: An additional property for a PercentagesValue
        object.

    Fields:
      additionalProperties: Additional properties of type PercentagesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a PercentagesValue object.

      Fields:
        key: Name of the additional property.
        value: A number attribute.
      """
        key = _messages.StringField(1)
        value = _messages.FloatField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)