from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ConditionsValue(_messages.Message):
    """Output only. The reason(s) why a trigger is in FAILED state.

    Messages:
      AdditionalProperty: An additional property for a ConditionsValue object.

    Fields:
      additionalProperties: Additional properties of type ConditionsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ConditionsValue object.

      Fields:
        key: Name of the additional property.
        value: A StateCondition attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('StateCondition', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)