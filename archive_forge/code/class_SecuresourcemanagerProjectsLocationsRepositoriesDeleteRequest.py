from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuresourcemanagerProjectsLocationsRepositoriesDeleteRequest(_messages.Message):
    """A SecuresourcemanagerProjectsLocationsRepositoriesDeleteRequest object.

  Fields:
    allowMissing: Optional. If set to true, and the repository is not found,
      the request will succeed but no action will be taken on the server.
    name: Required. Name of the repository to delete. The format is projects/{
      project_number}/locations/{location_id}/repositories/{repository_id}.
  """
    allowMissing = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)