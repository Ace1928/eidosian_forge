from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StructuredStorageInfo(_messages.Message):
    """StructuredStorageInfo contains details about the data stored in
  Structured Storage for the referenced resource.

  Fields:
    sizeBytes: Size in bytes of data stored in structured storage.
  """
    sizeBytes = _messages.IntegerField(1)