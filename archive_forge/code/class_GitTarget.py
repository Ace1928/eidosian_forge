from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GitTarget(_messages.Message):
    """A Git repository to be used as a deployment target.

  Fields:
    branch: Git branch.
    directory: Subdirectory inside the repository. Example: 'staging/my-
      package'
    repo: Repository URL. Example:
      'https://github.com/kubernetes/examples.git'
  """
    branch = _messages.StringField(1)
    directory = _messages.StringField(2)
    repo = _messages.StringField(3)