from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListMavenArtifactsResponse(_messages.Message):
    """The response from listing maven artifacts.

  Fields:
    mavenArtifacts: The maven artifacts returned.
    nextPageToken: The token to retrieve the next page of artifacts, or empty
      if there are no more artifacts to return.
  """
    mavenArtifacts = _messages.MessageField('MavenArtifact', 1, repeated=True)
    nextPageToken = _messages.StringField(2)