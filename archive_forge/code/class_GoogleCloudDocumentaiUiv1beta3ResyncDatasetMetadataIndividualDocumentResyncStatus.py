from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3ResyncDatasetMetadataIndividualDocumentResyncStatus(_messages.Message):
    """Resync status for each document per inconsistency type.

  Enums:
    DocumentInconsistencyTypeValueValuesEnum: The type of document
      inconsistency.

  Fields:
    documentId: The document identifier.
    documentInconsistencyType: The type of document inconsistency.
    status: The status of resyncing the document with regards to the detected
      inconsistency. Empty if ResyncDatasetRequest.validate_only is `true`.
  """

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
    documentId = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3DocumentId', 1)
    documentInconsistencyType = _messages.EnumField('DocumentInconsistencyTypeValueValuesEnum', 2)
    status = _messages.MessageField('GoogleRpcStatus', 3)