from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemoteworkersV1test2Directory(_messages.Message):
    """The contents of a directory. Similar to the equivalent message in the
  Remote Execution API.

  Fields:
    directories: Any subdirectories
    files: The files in this directory
  """
    directories = _messages.MessageField('GoogleDevtoolsRemoteworkersV1test2DirectoryMetadata', 1, repeated=True)
    files = _messages.MessageField('GoogleDevtoolsRemoteworkersV1test2FileMetadata', 2, repeated=True)