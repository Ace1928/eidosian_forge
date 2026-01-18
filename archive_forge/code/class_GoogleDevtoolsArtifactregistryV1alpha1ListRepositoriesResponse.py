from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsArtifactregistryV1alpha1ListRepositoriesResponse(_messages.Message):
    """The response from listing repositories.

  Fields:
    nextPageToken: The token to retrieve the next page of repositories, or
      empty if there are no more repositories to return.
    repositories: The repositories returned.
  """
    nextPageToken = _messages.StringField(1)
    repositories = _messages.MessageField('GoogleDevtoolsArtifactregistryV1alpha1Repository', 2, repeated=True)