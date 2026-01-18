from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigAptRepositoryArtifactRegistryRepository(_messages.Message):
    """A representation of an Artifact Registry repository.

  Fields:
    repository: A reference to the repository resource, for example:
      `projects/p1/locations/us-central1/repositories/repo1`.
  """
    repository = _messages.StringField(1)