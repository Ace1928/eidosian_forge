from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2AgentAssistantFeedbackSummarizationFeedback(_messages.Message):
    """Feedback for conversation summarization.

  Messages:
    TextSectionsValue: Optional. Actual text sections of submitted summary.

  Fields:
    startTime: Timestamp when composing of the summary starts.
    submitTime: Timestamp when the summary was submitted.
    summaryText: Text of actual submitted summary.
    textSections: Optional. Actual text sections of submitted summary.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TextSectionsValue(_messages.Message):
        """Optional. Actual text sections of submitted summary.

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
    startTime = _messages.StringField(1)
    submitTime = _messages.StringField(2)
    summaryText = _messages.StringField(3)
    textSections = _messages.MessageField('TextSectionsValue', 4)