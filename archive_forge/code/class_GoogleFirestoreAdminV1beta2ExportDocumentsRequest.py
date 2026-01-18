from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1beta2ExportDocumentsRequest(_messages.Message):
    """The request for FirestoreAdmin.ExportDocuments.

  Fields:
    collectionIds: Which collection ids to export. Unspecified means all
      collections.
    outputUriPrefix: The output URI. Currently only supports Google Cloud
      Storage URIs of the form: `gs://BUCKET_NAME[/NAMESPACE_PATH]`, where
      `BUCKET_NAME` is the name of the Google Cloud Storage bucket and
      `NAMESPACE_PATH` is an optional Google Cloud Storage namespace path.
      When choosing a name, be sure to consider Google Cloud Storage naming
      guidelines: https://cloud.google.com/storage/docs/naming. If the URI is
      a bucket (without a namespace path), a prefix will be generated based on
      the start time.
  """
    collectionIds = _messages.StringField(1, repeated=True)
    outputUriPrefix = _messages.StringField(2)