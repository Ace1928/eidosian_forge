from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2StorageSource(_messages.Message):
    """Location of the source in an archive file in Google Cloud Storage.

  Fields:
    bucket: Required. Google Cloud Storage bucket containing the source (see
      [Bucket Name Requirements](https://cloud.google.com/storage/docs/bucket-
      naming#requirements)).
    generation: Optional. Google Cloud Storage generation for the object. If
      the generation is omitted, the latest generation will be used.
    object: Required. Google Cloud Storage object containing the source. This
      object must be a gzipped archive file (`.tar.gz`) containing source to
      build.
  """
    bucket = _messages.StringField(1)
    generation = _messages.IntegerField(2)
    object = _messages.StringField(3)