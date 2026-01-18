from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1ImportDocumentsRequest(_messages.Message):
    """The request for FirestoreAdmin.ImportDocuments.

  Fields:
    collectionIds: Which collection ids to import. Unspecified means all
      collections included in the import.
    inputUriPrefix: Location of the exported files. This must match the
      output_uri_prefix of an ExportDocumentsResponse from an export that has
      completed successfully. See:
      google.firestore.admin.v1.ExportDocumentsResponse.output_uri_prefix.
    namespaceIds: An empty list represents all namespaces. This is the
      preferred usage for databases that don't use namespaces. An empty string
      element represents the default namespace. This should be used if the
      database has data in non-default namespaces, but doesn't want to include
      them. Each namespace in this list must be unique.
  """
    collectionIds = _messages.StringField(1, repeated=True)
    inputUriPrefix = _messages.StringField(2)
    namespaceIds = _messages.StringField(3, repeated=True)