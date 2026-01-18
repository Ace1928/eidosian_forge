from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class NinthValue(_messages.Message):
    """Map field.

    Messages:
      AdditionalProperty: An additional property for a NinthValue object.

    Fields:
      additionalProperties: Additional properties of type NinthValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a NinthValue object.

      Fields:
        key: Name of the additional property.
        value: A HelloWorldFooBar attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('HelloWorldFooBar', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)