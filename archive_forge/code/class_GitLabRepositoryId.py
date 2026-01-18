from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GitLabRepositoryId(_messages.Message):
    """GitLabRepositoryId identifies a specific repository hosted on GitLab.com
  or GitLabEnterprise

  Fields:
    id: Required. Identifier for the repository. example: "namespace/project-
      slug", namespace is usually the username or group ID
    webhookId: Output only. The ID of the webhook that was created for
      receiving events from this repo. We only create and manage a single
      webhook for each repo.
  """
    id = _messages.StringField(1)
    webhookId = _messages.IntegerField(2, variant=_messages.Variant.INT32)