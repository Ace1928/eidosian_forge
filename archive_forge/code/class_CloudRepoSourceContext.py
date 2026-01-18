from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudRepoSourceContext(_messages.Message):
    """A CloudRepoSourceContext denotes a particular revision in a Google Cloud
  Source Repo.

  Fields:
    aliasContext: An alias, which may be a branch or tag.
    repoId: The ID of the repo.
    revisionId: A revision ID.
  """
    aliasContext = _messages.MessageField('AliasContext', 1)
    repoId = _messages.MessageField('RepoId', 2)
    revisionId = _messages.StringField(3)