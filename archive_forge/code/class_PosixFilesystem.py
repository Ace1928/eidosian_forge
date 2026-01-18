from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PosixFilesystem(_messages.Message):
    """A POSIX filesystem resource.

  Fields:
    rootDirectory: Root directory path to the filesystem.
  """
    rootDirectory = _messages.StringField(1)