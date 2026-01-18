from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2GenerateStatelessSummaryResponseSummary(_messages.Message):
    """Generated summary for a conversation.

  Messages:
    TextSectionsValue: The summary content that is divided into sections. The
      key is the section's name and the value is the section's content. There
      is no specific format for the key or value.

  Fields:
    baselineModelVersion: The baseline model version used to generate this
      summary. It is empty if a baseline model was not used to generate this
      summary.
    text: The summary content that is concatenated into one string.
    textSections: The summary content that is divided into sections. The key
      is the section's name and the value is the section's content. There is
      no specific format for the key or value.
  """

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
    baselineModelVersion = _messages.StringField(1)
    text = _messages.StringField(2)
    textSections = _messages.MessageField('TextSectionsValue', 3)