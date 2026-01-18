from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class OperationParamsValue(_messages.Message):
    """Optional. Request parameters that will be used for executing this
    operation. The struct should be in a form of map with param name as the
    key and actual param value as the value. E.g. If this operation requires a
    param "name" to be set to "abc". you can set this to something like
    {"name": "abc"}.

    Messages:
      AdditionalProperty: An additional property for a OperationParamsValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a OperationParamsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('extra_types.JsonValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)