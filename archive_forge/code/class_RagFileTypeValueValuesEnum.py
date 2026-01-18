from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RagFileTypeValueValuesEnum(_messages.Enum):
    """Output only. The type of the RagFile.

    Values:
      RAG_FILE_TYPE_UNSPECIFIED: RagFile type is unspecified.
      RAG_FILE_TYPE_TXT: RagFile type is TXT.
      RAG_FILE_TYPE_PDF: RagFile type is PDF.
    """
    RAG_FILE_TYPE_UNSPECIFIED = 0
    RAG_FILE_TYPE_TXT = 1
    RAG_FILE_TYPE_PDF = 2