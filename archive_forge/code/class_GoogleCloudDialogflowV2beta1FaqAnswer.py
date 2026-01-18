from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1FaqAnswer(_messages.Message):
    """Represents answer from "frequently asked questions".

  Messages:
    MetadataValue: A map that contains metadata about the answer and the
      document from which it originates.

  Fields:
    answer: The piece of text from the `source` knowledge base document.
    answerRecord: The name of answer record, in the format of
      "projects//locations//answerRecords/"
    confidence: The system's confidence score that this Knowledge answer is a
      good match for this conversational query, range from 0.0 (completely
      uncertain) to 1.0 (completely certain).
    metadata: A map that contains metadata about the answer and the document
      from which it originates.
    question: The corresponding FAQ question.
    source: Indicates which Knowledge Document this answer was extracted from.
      Format: `projects//locations//agent/knowledgeBases//documents/`.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """A map that contains metadata about the answer and the document from
    which it originates.

    Messages:
      AdditionalProperty: An additional property for a MetadataValue object.

    Fields:
      additionalProperties: Additional properties of type MetadataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    answer = _messages.StringField(1)
    answerRecord = _messages.StringField(2)
    confidence = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    metadata = _messages.MessageField('MetadataValue', 4)
    question = _messages.StringField(5)
    source = _messages.StringField(6)