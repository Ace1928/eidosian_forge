from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class CausesValue(_messages.Message):
    """The straggler causes, keyed by the string representation of the
    StragglerCause enum and contains specialized debugging information for
    each straggler cause.

    Messages:
      AdditionalProperty: An additional property for a CausesValue object.

    Fields:
      additionalProperties: Additional properties of type CausesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a CausesValue object.

      Fields:
        key: Name of the additional property.
        value: A StragglerDebuggingInfo attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('StragglerDebuggingInfo', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)