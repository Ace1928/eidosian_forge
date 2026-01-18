from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchDeleteVersionsMetadata(_messages.Message):
    """The metadata of an LRO from deleting multiple versions.

  Fields:
    failedVersions: The versions the operation failed to delete.
  """
    failedVersions = _messages.StringField(1, repeated=True)