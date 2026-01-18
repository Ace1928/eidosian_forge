from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class DiagnosticInfoValue(_messages.Message):
    """Free-form diagnostic information for the associated detect intent
    request. The fields of this data can change without notice, so you should
    not write code that depends on its structure. The data may contain: -
    webhook call latency - webhook errors

    Messages:
      AdditionalProperty: An additional property for a DiagnosticInfoValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a DiagnosticInfoValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('extra_types.JsonValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)