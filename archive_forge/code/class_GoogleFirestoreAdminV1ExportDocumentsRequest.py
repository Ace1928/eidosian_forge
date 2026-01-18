from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1ExportDocumentsRequest(_messages.Message):
    """The request for FirestoreAdmin.ExportDocuments.

  Fields:
    collectionIds: Which collection ids to export. Unspecified means all
      collections.
    namespaceIds: An empty list represents all namespaces. This is the
      preferred usage for databases that don't use namespaces. An empty string
      element represents the default namespace. This should be used if the
      database has data in non-default namespaces, but doesn't want to include
      them. Each namespace in this list must be unique.
    outputUriPrefix: The output URI. Currently only supports Google Cloud
      Storage URIs of the form: `gs://BUCKET_NAME[/NAMESPACE_PATH]`, where
      `BUCKET_NAME` is the name of the Google Cloud Storage bucket and
      `NAMESPACE_PATH` is an optional Google Cloud Storage namespace path.
      When choosing a name, be sure to consider Google Cloud Storage naming
      guidelines: https://cloud.google.com/storage/docs/naming. If the URI is
      a bucket (without a namespace path), a prefix will be generated based on
      the start time.
    snapshotTime: The timestamp that corresponds to the version of the
      database to be exported. The timestamp must be in the past, rounded to
      the minute and not older than earliestVersionTime. If specified, then
      the exported documents will represent a consistent view of the database
      at the provided time. Otherwise, there are no guarantees about the
      consistency of the exported documents.
  """
    collectionIds = _messages.StringField(1, repeated=True)
    namespaceIds = _messages.StringField(2, repeated=True)
    outputUriPrefix = _messages.StringField(3)
    snapshotTime = _messages.StringField(4)