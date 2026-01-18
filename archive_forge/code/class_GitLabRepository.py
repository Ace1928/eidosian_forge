from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GitLabRepository(_messages.Message):
    """Proto Representing a GitLabRepository

  Fields:
    browseUri: Link to the browse repo page on the GitLab instance
    description: Description of the repository
    displayName: Display name of the repository
    name: The resource name of the repository
    repositoryId: Identifier for a repository
  """
    browseUri = _messages.StringField(1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    name = _messages.StringField(4)
    repositoryId = _messages.MessageField('GitLabRepositoryId', 5)