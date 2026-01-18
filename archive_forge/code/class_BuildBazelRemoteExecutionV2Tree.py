from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2Tree(_messages.Message):
    """A `Tree` contains all the Directory protos in a single directory Merkle
  tree, compressed into one message.

  Fields:
    children: All the child directories: the directories referred to by the
      root and, recursively, all its children. In order to reconstruct the
      directory tree, the client must take the digests of each of the child
      directories and then build up a tree starting from the `root`. Servers
      SHOULD ensure that these are ordered consistently such that two actions
      producing equivalent output directories on the same server
      implementation also produce Tree messages with matching digests.
    root: The root directory in the tree.
  """
    children = _messages.MessageField('BuildBazelRemoteExecutionV2Directory', 1, repeated=True)
    root = _messages.MessageField('BuildBazelRemoteExecutionV2Directory', 2)