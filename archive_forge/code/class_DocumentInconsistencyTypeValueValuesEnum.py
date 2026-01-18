from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentInconsistencyTypeValueValuesEnum(_messages.Enum):
    """The type of document inconsistency.

    Values:
      DOCUMENT_INCONSISTENCY_TYPE_UNSPECIFIED: Default value.
      DOCUMENT_INCONSISTENCY_TYPE_INVALID_DOCPROTO: The document proto is
        invalid.
      DOCUMENT_INCONSISTENCY_TYPE_MISMATCHED_METADATA: Indexed docproto
        metadata is mismatched.
      DOCUMENT_INCONSISTENCY_TYPE_NO_PAGE_IMAGE: The page image or thumbnails
        are missing.
    """
    DOCUMENT_INCONSISTENCY_TYPE_UNSPECIFIED = 0
    DOCUMENT_INCONSISTENCY_TYPE_INVALID_DOCPROTO = 1
    DOCUMENT_INCONSISTENCY_TYPE_MISMATCHED_METADATA = 2
    DOCUMENT_INCONSISTENCY_TYPE_NO_PAGE_IMAGE = 3