from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1ArticleAnswer(_messages.Message):
    """Represents article answer.

  Messages:
    MetadataValue: A map that contains metadata about the answer and the
      document from which it originates.

  Fields:
    answerRecord: The name of answer record, in the format of
      "projects//locations//answerRecords/"
    metadata: A map that contains metadata about the answer and the document
      from which it originates.
    snippets: Output only. Article snippets.
    title: The article title.
    uri: The article URI.
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
    answerRecord = _messages.StringField(1)
    metadata = _messages.MessageField('MetadataValue', 2)
    snippets = _messages.StringField(3, repeated=True)
    title = _messages.StringField(4)
    uri = _messages.StringField(5)