from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListRepositoriesResponse(_messages.Message):
    """Message for response to listing Repositories.

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    repositories: The list of Repositories.
  """
    nextPageToken = _messages.StringField(1)
    repositories = _messages.MessageField('Repository', 2, repeated=True)