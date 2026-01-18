from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileHashes(_messages.Message):
    """Container message for hashes of byte content of files, used in source
  messages to verify integrity of source input to the build.

  Fields:
    fileHash: Required. Collection of file hashes.
  """
    fileHash = _messages.MessageField('Hash', 1, repeated=True)