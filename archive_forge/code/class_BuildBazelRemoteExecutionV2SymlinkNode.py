from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2SymlinkNode(_messages.Message):
    """A `SymlinkNode` represents a symbolic link.

  Fields:
    name: The name of the symlink.
    nodeProperties: A BuildBazelRemoteExecutionV2NodeProperties attribute.
    target: The target path of the symlink. The path separator is a forward
      slash `/`. The target path can be relative to the parent directory of
      the symlink or it can be an absolute path starting with `/`. Support for
      absolute paths can be checked using the Capabilities API. `..`
      components are allowed anywhere in the target path as logical
      canonicalization may lead to different behavior in the presence of
      directory symlinks (e.g. `foo/../bar` may not be the same as `bar`). To
      reduce potential cache misses, canonicalization is still recommended
      where this is possible without impacting correctness.
  """
    name = _messages.StringField(1)
    nodeProperties = _messages.MessageField('BuildBazelRemoteExecutionV2NodeProperties', 2)
    target = _messages.StringField(3)