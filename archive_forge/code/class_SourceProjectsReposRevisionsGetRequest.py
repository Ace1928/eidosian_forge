from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceProjectsReposRevisionsGetRequest(_messages.Message):
    """A SourceProjectsReposRevisionsGetRequest object.

  Fields:
    projectId: The ID of the project.
    repoId_uid: A server-assigned, globally unique identifier.
    repoName: The name of the repo. Leave empty for the default repo.
    revisionId: The ID of the revision.
  """
    projectId = _messages.StringField(1, required=True)
    repoId_uid = _messages.StringField(2)
    repoName = _messages.StringField(3, required=True)
    revisionId = _messages.StringField(4, required=True)