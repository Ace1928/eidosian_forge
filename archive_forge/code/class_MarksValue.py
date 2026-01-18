from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class MarksValue(_messages.Message):
    """Mutable user specified security marks belonging to the parent
    resource. Constraints are as follows: * Keys and values are treated as
    case insensitive * Keys must be between 1 - 256 characters (inclusive) *
    Keys must be letters, numbers, underscores, or dashes * Values have
    leading and trailing whitespace trimmed, remaining characters must be
    between 1 - 4096 characters (inclusive)

    Messages:
      AdditionalProperty: An additional property for a MarksValue object.

    Fields:
      additionalProperties: Additional properties of type MarksValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a MarksValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)