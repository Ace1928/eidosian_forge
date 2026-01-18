from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceProjectsReposAliasesUpdateRequest(_messages.Message):
    """A SourceProjectsReposAliasesUpdateRequest object.

  Fields:
    alias: A Alias resource to be passed as the request body.
    aliasesId: A string attribute.
    oldRevisionId: If non-empty, must match the revision that the alias refers
      to.
    projectId: The ID of the project.
    repoId_uid: A server-assigned, globally unique identifier.
    repoName: The name of the repo. Leave empty for the default repo.
  """
    alias = _messages.MessageField('Alias', 1)
    aliasesId = _messages.StringField(2, required=True)
    oldRevisionId = _messages.StringField(3)
    projectId = _messages.StringField(4, required=True)
    repoId_uid = _messages.StringField(5)
    repoName = _messages.StringField(6, required=True)