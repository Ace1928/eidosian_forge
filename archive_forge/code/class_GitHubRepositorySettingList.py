from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GitHubRepositorySettingList(_messages.Message):
    """A wrapper message for a list of GitHubRepositorySettings.

  Fields:
    repositorySettings: A list of GitHubRepositorySettings.
  """
    repositorySettings = _messages.MessageField('GitHubRepositorySetting', 1, repeated=True)