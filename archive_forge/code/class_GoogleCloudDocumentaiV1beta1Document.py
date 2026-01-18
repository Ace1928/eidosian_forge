from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta1Document(_messages.Message):
    """Document represents the canonical document resource in Document AI. It
  is an interchange format that provides insights into documents and allows
  for collaboration between users and Document AI to iterate and optimize for
  quality.

  Fields:
    content: Optional. Inline document content, represented as a stream of
      bytes. Note: As with all `bytes` fields, protobuffers use a pure binary
      representation, whereas JSON representations use base64.
    entities: A list of entities detected on Document.text. For document
      shards, entities in this list may cross shard boundaries.
    entityRelations: Placeholder. Relationship among Document.entities.
    error: Any error that occurred while processing this document.
    mimeType: An IANA published [media type (MIME
      type)](https://www.iana.org/assignments/media-types/media-types.xhtml).
    pages: Visual page layout for the Document.
    revisions: Placeholder. Revision history of this document.
    shardInfo: Information about the sharding if this document is sharded part
      of a larger document. If the document is not sharded, this message is
      not specified.
    text: Optional. UTF-8 encoded text in reading order from the document.
    textChanges: Placeholder. A list of text corrections made to
      Document.text. This is usually used for annotating corrections to OCR
      mistakes. Text changes for a given revision may not overlap with each
      other.
    textStyles: Styles for the Document.text.
    uri: Optional. Currently supports Google Cloud Storage URI of the form
      `gs://bucket_name/object_name`. Object versioning is not supported. For
      more information, refer to [Google Cloud Storage Request
      URIs](https://cloud.google.com/storage/docs/reference-uris).
  """
    content = _messages.BytesField(1)
    entities = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentEntity', 2, repeated=True)
    entityRelations = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentEntityRelation', 3, repeated=True)
    error = _messages.MessageField('GoogleRpcStatus', 4)
    mimeType = _messages.StringField(5)
    pages = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentPage', 6, repeated=True)
    revisions = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentRevision', 7, repeated=True)
    shardInfo = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentShardInfo', 8)
    text = _messages.StringField(9)
    textChanges = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentTextChange', 10, repeated=True)
    textStyles = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentStyle', 11, repeated=True)
    uri = _messages.StringField(12)