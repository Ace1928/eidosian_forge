from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2DirectoryNode(_messages.Message):
    """A `DirectoryNode` represents a child of a Directory which is itself a
  `Directory` and its associated metadata.

  Fields:
    digest: The digest of the Directory object represented. See Digest for
      information about how to take the digest of a proto message.
    name: The name of the directory.
  """
    digest = _messages.MessageField('BuildBazelRemoteExecutionV2Digest', 1)
    name = _messages.StringField(2)