from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BitbucketServerRepositoryId(_messages.Message):
    """BitbucketServerRepositoryId identifies a specific repository hosted on a
  Bitbucket Server.

  Fields:
    projectKey: Required. Identifier for the project storing the repository.
    repoSlug: Required. Identifier for the repository.
    webhookId: Output only. The ID of the webhook that was created for
      receiving events from this repo. We only create and manage a single
      webhook for each repo.
  """
    projectKey = _messages.StringField(1)
    repoSlug = _messages.StringField(2)
    webhookId = _messages.IntegerField(3, variant=_messages.Variant.INT32)