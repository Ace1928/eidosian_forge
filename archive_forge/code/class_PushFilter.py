from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PushFilter(_messages.Message):
    """Push contains filter properties for matching GitHub git pushes.

  Fields:
    branch: Regexes matching branches to build. The syntax of the regular
      expressions accepted is the syntax accepted by RE2 and described at
      https://github.com/google/re2/wiki/Syntax
    invertRegex: When true, only trigger a build if the revision regex does
      NOT match the git_ref regex.
    tag: Regexes matching tags to build. The syntax of the regular expressions
      accepted is the syntax accepted by RE2 and described at
      https://github.com/google/re2/wiki/Syntax
  """
    branch = _messages.StringField(1)
    invertRegex = _messages.BooleanField(2)
    tag = _messages.StringField(3)