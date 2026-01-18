from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
def GetExportDocumentsRequest(database, output_uri_prefix, namespace_ids=None, collection_ids=None, snapshot_time=None):
    """Returns a request for a Firestore Admin Export.

  Args:
    database: the database id to export, a string.
    output_uri_prefix: the output GCS path prefix, a string.
    namespace_ids: a string list of namespace ids to export.
    collection_ids: a string list of collection ids to export.
    snapshot_time: the version of the database to export, as string in
      google-datetime format.

  Returns:
    an ExportDocumentsRequest message.
  """
    messages = api_utils.GetMessages()
    export_request = messages.GoogleFirestoreAdminV1ExportDocumentsRequest(outputUriPrefix=output_uri_prefix, namespaceIds=namespace_ids if namespace_ids else [], collectionIds=collection_ids if collection_ids else [], snapshotTime=snapshot_time)
    request = messages.FirestoreProjectsDatabasesExportDocumentsRequest(name=database, googleFirestoreAdminV1ExportDocumentsRequest=export_request)
    return request