from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesReindexRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesReindexRequest object.

  Fields:
    name: Required. The name of the repository to refresh.
    reindexRepositoryRequest: A ReindexRepositoryRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    reindexRepositoryRequest = _messages.MessageField('ReindexRepositoryRequest', 2)