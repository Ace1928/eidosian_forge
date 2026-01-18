from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1GcsDestination(_messages.Message):
    """A Cloud Storage location.

  Fields:
    uri: Required. The URI of the Cloud Storage object. It's the same URI that
      is used by gsutil. Example: "gs://bucket_name/object_name". See [Viewing
      and Editing Object
      Metadata](https://cloud.google.com/storage/docs/viewing-editing-
      metadata) for more information. If the specified Cloud Storage object
      already exists and there is no
      [hold](https://cloud.google.com/storage/docs/object-holds), it will be
      overwritten with the analysis result.
  """
    uri = _messages.StringField(1)