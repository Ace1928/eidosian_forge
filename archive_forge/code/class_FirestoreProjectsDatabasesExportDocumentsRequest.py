from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesExportDocumentsRequest(_messages.Message):
    """A FirestoreProjectsDatabasesExportDocumentsRequest object.

  Fields:
    googleFirestoreAdminV1ExportDocumentsRequest: A
      GoogleFirestoreAdminV1ExportDocumentsRequest resource to be passed as
      the request body.
    name: Required. Database to export. Should be of the form:
      `projects/{project_id}/databases/{database_id}`.
  """
    googleFirestoreAdminV1ExportDocumentsRequest = _messages.MessageField('GoogleFirestoreAdminV1ExportDocumentsRequest', 1)
    name = _messages.StringField(2, required=True)