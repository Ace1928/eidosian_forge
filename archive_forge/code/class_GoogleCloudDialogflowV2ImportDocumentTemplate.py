from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ImportDocumentTemplate(_messages.Message):
    """The template used for importing documents.

  Enums:
    KnowledgeTypesValueListEntryValuesEnum:

  Messages:
    MetadataValue: Metadata for the document. The metadata supports arbitrary
      key-value pairs. Suggested use cases include storing a document's title,
      an external URL distinct from the document's content_uri, etc. The max
      size of a `key` or a `value` of the metadata is 1024 bytes.

  Fields:
    knowledgeTypes: Required. The knowledge type of document content.
    metadata: Metadata for the document. The metadata supports arbitrary key-
      value pairs. Suggested use cases include storing a document's title, an
      external URL distinct from the document's content_uri, etc. The max size
      of a `key` or a `value` of the metadata is 1024 bytes.
    mimeType: Required. The MIME type of the document.
  """

    class KnowledgeTypesValueListEntryValuesEnum(_messages.Enum):
        """KnowledgeTypesValueListEntryValuesEnum enum type.

    Values:
      KNOWLEDGE_TYPE_UNSPECIFIED: The type is unspecified or arbitrary.
      FAQ: The document content contains question and answer pairs as either
        HTML or CSV. Typical FAQ HTML formats are parsed accurately, but
        unusual formats may fail to be parsed. CSV must have questions in the
        first column and answers in the second, with no header. Because of
        this explicit format, they are always parsed accurately.
      EXTRACTIVE_QA: Documents for which unstructured text is extracted and
        used for question answering.
      ARTICLE_SUGGESTION: The entire document content as a whole can be used
        for query results. Only for Contact Center Solutions on Dialogflow.
      AGENT_FACING_SMART_REPLY: The document contains agent-facing Smart Reply
        entries.
    """
        KNOWLEDGE_TYPE_UNSPECIFIED = 0
        FAQ = 1
        EXTRACTIVE_QA = 2
        ARTICLE_SUGGESTION = 3
        AGENT_FACING_SMART_REPLY = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """Metadata for the document. The metadata supports arbitrary key-value
    pairs. Suggested use cases include storing a document's title, an external
    URL distinct from the document's content_uri, etc. The max size of a `key`
    or a `value` of the metadata is 1024 bytes.

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
    knowledgeTypes = _messages.EnumField('KnowledgeTypesValueListEntryValuesEnum', 1, repeated=True)
    metadata = _messages.MessageField('MetadataValue', 2)
    mimeType = _messages.StringField(3)