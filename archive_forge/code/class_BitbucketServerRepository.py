from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BitbucketServerRepository(_messages.Message):
    """BitbucketServerRepository represents a repository hosted on a Bitbucket
  Server.

  Fields:
    browseUri: Link to the browse repo page on the Bitbucket Server instance.
    description: Description of the repository.
    displayName: Display name of the repository.
    name: The resource name of the repository.
    repoId: Identifier for a repository hosted on a Bitbucket Server.
  """
    browseUri = _messages.StringField(1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    name = _messages.StringField(4)
    repoId = _messages.MessageField('BitbucketServerRepositoryId', 5)