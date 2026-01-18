from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GitSource(_messages.Message):
    """A set of files in a Git repository.

  Fields:
    directory: Optional. Subdirectory inside the repository. Example:
      'staging/my-package'
    ref: Optional. Git reference (e.g. branch or tag).
    repo: Optional. Repository URL. Example:
      'https://github.com/kubernetes/examples.git'
  """
    directory = _messages.StringField(1)
    ref = _messages.StringField(2)
    repo = _messages.StringField(3)