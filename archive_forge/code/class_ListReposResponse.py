from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListReposResponse(_messages.Message):
    """Response for ListRepos. The size is not set in the returned
  repositories.

  Fields:
    nextPageToken: If non-empty, additional repositories exist within the
      project. These can be retrieved by including this value in the next
      ListReposRequest's page_token field.
    repos: The listed repos.
  """
    nextPageToken = _messages.StringField(1)
    repos = _messages.MessageField('Repo', 2, repeated=True)