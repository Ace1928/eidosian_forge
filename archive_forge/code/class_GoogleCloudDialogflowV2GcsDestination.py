from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2GcsDestination(_messages.Message):
    """Google Cloud Storage location for the output.

  Fields:
    uri: The Google Cloud Storage URIs for the output. A URI is of the form:
      `gs://bucket/object-prefix-or-name` Whether a prefix or name is used
      depends on the use case. The requesting user must have "write-
      permission" to the bucket.
  """
    uri = _messages.StringField(1)