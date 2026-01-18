from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2OutputSymlink(_messages.Message):
    """An `OutputSymlink` is similar to a Symlink, but it is used as an output
  in an `ActionResult`. `OutputSymlink` is binary-compatible with
  `SymlinkNode`.

  Fields:
    nodeProperties: A BuildBazelRemoteExecutionV2NodeProperties attribute.
    path: The full path of the symlink relative to the working directory,
      including the filename. The path separator is a forward slash `/`. Since
      this is a relative path, it MUST NOT begin with a leading forward slash.
    target: The target path of the symlink. The path separator is a forward
      slash `/`. The target path can be relative to the parent directory of
      the symlink or it can be an absolute path starting with `/`. Support for
      absolute paths can be checked using the Capabilities API. `..`
      components are allowed anywhere in the target path.
  """
    nodeProperties = _messages.MessageField('BuildBazelRemoteExecutionV2NodeProperties', 1)
    path = _messages.StringField(2)
    target = _messages.StringField(3)