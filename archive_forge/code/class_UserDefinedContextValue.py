from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class UserDefinedContextValue(_messages.Message):
    """User-defined context as a set of key/value pairs, which will be sent
    as function invocation context together with batched arguments in the
    requests to the remote service. The total number of bytes of keys and
    values must be less than 8KB.

    Messages:
      AdditionalProperty: An additional property for a UserDefinedContextValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        UserDefinedContextValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a UserDefinedContextValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)