from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentOutputConfigGcsOutputConfig(_messages.Message):
    """The configuration used when outputting documents.

  Fields:
    fieldMask: Specifies which fields to include in the output documents. Only
      supports top level document and pages field so it must be in the form of
      `{document_field_name}` or `pages.{page_field_name}`.
    gcsUri: The Cloud Storage uri (a directory) of the output.
    shardingConfig: Specifies the sharding config for the output document.
  """
    fieldMask = _messages.StringField(1)
    gcsUri = _messages.StringField(2)
    shardingConfig = _messages.MessageField('GoogleCloudDocumentaiV1DocumentOutputConfigGcsOutputConfigShardingConfig', 3)