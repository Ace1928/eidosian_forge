from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceProjectsReposUpdateRequest(_messages.Message):
    """A SourceProjectsReposUpdateRequest object.

  Fields:
    projectId: The ID of the project.
    repoName: The name of the repo. Leave empty for the default repo.
    updateRepoRequest: A UpdateRepoRequest resource to be passed as the
      request body.
  """
    projectId = _messages.StringField(1, required=True)
    repoName = _messages.StringField(2, required=True)
    updateRepoRequest = _messages.MessageField('UpdateRepoRequest', 3)