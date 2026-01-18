from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemoteworkersV1test2DirectoryMetadata(_messages.Message):
    """The metadata for a directory. Similar to the equivalent message in the
  Remote Execution API.

  Fields:
    digest: A pointer to the contents of the directory, in the form of a
      marshalled Directory message.
    path: The path of the directory, as in FileMetadata.path.
  """
    digest = _messages.MessageField('GoogleDevtoolsRemoteworkersV1test2Digest', 1)
    path = _messages.StringField(2)