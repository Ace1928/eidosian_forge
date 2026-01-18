from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1GcsPrefix(_messages.Message):
    """Specifies all documents on Cloud Storage with a common prefix.

  Fields:
    gcsUriPrefix: The URI prefix.
  """
    gcsUriPrefix = _messages.StringField(1)