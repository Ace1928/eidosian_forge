from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesPackagesVersionsDeleteRequest(_messages.Message):
    """A
  ArtifactregistryProjectsLocationsRepositoriesPackagesVersionsDeleteRequest
  object.

  Fields:
    force: By default, a version that is tagged may not be deleted. If
      force=true, the version and any tags pointing to the version are
      deleted.
    name: The name of the version to delete.
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)