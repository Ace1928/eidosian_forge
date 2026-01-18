from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesPackagesGetRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesPackagesGetRequest
  object.

  Fields:
    name: Required. The name of the package to retrieve.
  """
    name = _messages.StringField(1, required=True)