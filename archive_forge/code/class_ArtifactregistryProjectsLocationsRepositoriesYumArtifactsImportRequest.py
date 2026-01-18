from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesYumArtifactsImportRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesYumArtifactsImportRequest
  object.

  Fields:
    importYumArtifactsRequest: A ImportYumArtifactsRequest resource to be
      passed as the request body.
    parent: The name of the parent resource where the artifacts will be
      imported.
  """
    importYumArtifactsRequest = _messages.MessageField('ImportYumArtifactsRequest', 1)
    parent = _messages.StringField(2, required=True)