from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2Document(_messages.Message):
    """A knowledge document to be used by a KnowledgeBase. For more
  information, see the [knowledge base
  guide](https://cloud.google.com/dialogflow/docs/how/knowledge-bases). Note:
  The `projects.agent.knowledgeBases.documents` resource is deprecated; only
  use `projects.knowledgeBases.documents`.

  Enums:
    KnowledgeTypesValueListEntryValuesEnum:
    StateValueValuesEnum: Output only. The current state of the document.

  Messages:
    MetadataValue: Optional. Metadata for the document. The metadata supports
      arbitrary key-value pairs. Suggested use cases include storing a
      document's title, an external URL distinct from the document's
      content_uri, etc. The max size of a `key` or a `value` of the metadata
      is 1024 bytes.

  Fields:
    contentUri: The URI where the file content is located. For documents
      stored in Google Cloud Storage, these URIs must have the form `gs:///`.
      NOTE: External URLs must correspond to public webpages, i.e., they must
      be indexed by Google Search. In particular, URLs for showing documents
      in Google Cloud Storage (i.e. the URL in your browser) are not
      supported. Instead use the `gs://` format URI described above.
    displayName: Required. The display name of the document. The name must be
      1024 bytes or less; otherwise, the creation request fails.
    enableAutoReload: Optional. If true, we try to automatically reload the
      document every day (at a time picked by the system). If false or
      unspecified, we don't try to automatically reload the document.
      Currently you can only enable automatic reload for documents sourced
      from a public url, see `source` field for the source types. Reload
      status can be tracked in `latest_reload_status`. If a reload fails, we
      will keep the document unchanged. If a reload fails with internal
      errors, the system will try to reload the document on the next day. If a
      reload fails with non-retriable errors (e.g. PERMISSION_DENIED), the
      system will not try to reload the document anymore. You need to manually
      reload the document successfully by calling `ReloadDocument` and clear
      the errors.
    knowledgeTypes: Required. The knowledge type of document content.
    latestReloadStatus: Output only. The time and status of the latest reload.
      This reload may have been triggered automatically or manually and may
      not have succeeded.
    metadata: Optional. Metadata for the document. The metadata supports
      arbitrary key-value pairs. Suggested use cases include storing a
      document's title, an external URL distinct from the document's
      content_uri, etc. The max size of a `key` or a `value` of the metadata
      is 1024 bytes.
    mimeType: Required. The MIME type of this document.
    name: Optional. The document resource name. The name must be empty when
      creating a document. Format:
      `projects//locations//knowledgeBases//documents/`.
    rawContent: The raw content of the document. This field is only permitted
      for EXTRACTIVE_QA and FAQ knowledge types.
    state: Output only. The current state of the document.
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

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the document.

    Values:
      STATE_UNSPECIFIED: The document state is unspecified.
      CREATING: The document creation is in progress.
      ACTIVE: The document is active and ready to use.
      UPDATING: The document updation is in progress.
      RELOADING: The document is reloading.
      DELETING: The document deletion is in progress.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        UPDATING = 3
        RELOADING = 4
        DELETING = 5

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """Optional. Metadata for the document. The metadata supports arbitrary
    key-value pairs. Suggested use cases include storing a document's title,
    an external URL distinct from the document's content_uri, etc. The max
    size of a `key` or a `value` of the metadata is 1024 bytes.

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
    contentUri = _messages.StringField(1)
    displayName = _messages.StringField(2)
    enableAutoReload = _messages.BooleanField(3)
    knowledgeTypes = _messages.EnumField('KnowledgeTypesValueListEntryValuesEnum', 4, repeated=True)
    latestReloadStatus = _messages.MessageField('GoogleCloudDialogflowV2DocumentReloadStatus', 5)
    metadata = _messages.MessageField('MetadataValue', 6)
    mimeType = _messages.StringField(7)
    name = _messages.StringField(8)
    rawContent = _messages.BytesField(9)
    state = _messages.EnumField('StateValueValuesEnum', 10)