from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiNlLlmProtoServicePartDocumentMetadata(_messages.Message):
    """Metadata describes the original input document content.

  Fields:
    originalDocumentBlob: The original document blob.
    pageNumber: The (1-indexed) page number of the image in the original
      document. The first page carries the original document content and mime
      type.
  """
    originalDocumentBlob = _messages.MessageField('CloudAiNlLlmProtoServicePartBlob', 1)
    pageNumber = _messages.IntegerField(2, variant=_messages.Variant.INT32)