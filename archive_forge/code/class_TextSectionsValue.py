from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class TextSectionsValue(_messages.Message):
    """The summary content that is divided into sections. The key is the
    section's name and the value is the section's content. There is no
    specific format for the key or value.

    Messages:
      AdditionalProperty: An additional property for a TextSectionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type TextSectionsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a TextSectionsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)