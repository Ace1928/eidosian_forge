from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GcsDestination(_messages.Message):
    """A Cloud Storage location.

  Fields:
    uri: The URI of the Cloud Storage object. It's the same URI that is used
      by gsutil. Example: "gs://bucket_name/object_name". See [Viewing and
      Editing Object Metadata](https://cloud.google.com/storage/docs/viewing-
      editing-metadata) for more information. If the specified Cloud Storage
      object already exists and there is no
      [hold](https://cloud.google.com/storage/docs/object-holds), it will be
      overwritten with the exported result.
    uriPrefix: The URI prefix of all generated Cloud Storage objects. Example:
      "gs://bucket_name/object_name_prefix". Each object URI is in format:
      "gs://bucket_name/object_name_prefix// and only contains assets for that
      type. starts from 0. Example:
      "gs://bucket_name/object_name_prefix/compute.googleapis.com/Disk/0" is
      the first shard of output objects containing all
      compute.googleapis.com/Disk assets. An INVALID_ARGUMENT error will be
      returned if file with the same name
      "gs://bucket_name/object_name_prefix" already exists.
  """
    uri = _messages.StringField(1)
    uriPrefix = _messages.StringField(2)