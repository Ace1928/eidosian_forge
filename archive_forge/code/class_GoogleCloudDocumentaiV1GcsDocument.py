from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1GcsDocument(_messages.Message):
    """Specifies a document stored on Cloud Storage.

  Fields:
    gcsUri: The Cloud Storage object uri.
    mimeType: An IANA MIME type (RFC6838) of the content.
  """
    gcsUri = _messages.StringField(1)
    mimeType = _messages.StringField(2)