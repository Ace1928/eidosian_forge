from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SkaffoldGitSource(_messages.Message):
    """Git repository containing Skaffold Config modules.

  Fields:
    path: Optional. Relative path from the repository root to the Skaffold
      file.
    ref: Optional. Git ref the package should be cloned from.
    repo: Required. Git repository the package should be cloned from.
  """
    path = _messages.StringField(1)
    ref = _messages.StringField(2)
    repo = _messages.StringField(3)