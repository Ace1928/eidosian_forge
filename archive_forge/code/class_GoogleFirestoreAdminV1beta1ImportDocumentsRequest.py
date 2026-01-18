from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1beta1ImportDocumentsRequest(_messages.Message):
    """The request for FirestoreAdmin.ImportDocuments.

  Fields:
    collectionIds: Which collection ids to import. Unspecified means all
      collections included in the import.
    inputUriPrefix: Location of the exported files. This must match the
      output_uri_prefix of an ExportDocumentsResponse from an export that has
      completed successfully. See: google.firestore.admin.v1beta1.ExportDocume
      ntsResponse.output_uri_prefix.
  """
    collectionIds = _messages.StringField(1, repeated=True)
    inputUriPrefix = _messages.StringField(2)