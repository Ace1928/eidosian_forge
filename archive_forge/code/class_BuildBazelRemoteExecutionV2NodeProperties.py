from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2NodeProperties(_messages.Message):
    """Node properties for FileNodes, DirectoryNodes, and SymlinkNodes. The
  server is responsible for specifying the properties that it accepts.

  Fields:
    mtime: The file's last modification timestamp.
    properties: A list of string-based NodeProperties.
    unixMode: The UNIX file mode, e.g., 0755.
  """
    mtime = _messages.StringField(1)
    properties = _messages.MessageField('BuildBazelRemoteExecutionV2NodeProperty', 2, repeated=True)
    unixMode = _messages.IntegerField(3, variant=_messages.Variant.UINT32)