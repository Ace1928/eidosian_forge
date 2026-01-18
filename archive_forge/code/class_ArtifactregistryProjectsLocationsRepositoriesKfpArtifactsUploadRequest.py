from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesKfpArtifactsUploadRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesKfpArtifactsUploadRequest
  object.

  Fields:
    parent: The resource name of the repository where the KFP artifact will be
      uploaded.
    uploadKfpArtifactRequest: A UploadKfpArtifactRequest resource to be passed
      as the request body.
  """
    parent = _messages.StringField(1, required=True)
    uploadKfpArtifactRequest = _messages.MessageField('UploadKfpArtifactRequest', 2)