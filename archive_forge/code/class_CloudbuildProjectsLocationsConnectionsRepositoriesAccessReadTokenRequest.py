from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsConnectionsRepositoriesAccessReadTokenRequest(_messages.Message):
    """A
  CloudbuildProjectsLocationsConnectionsRepositoriesAccessReadTokenRequest
  object.

  Fields:
    fetchReadTokenRequest: A FetchReadTokenRequest resource to be passed as
      the request body.
    repository: Required. The resource name of the repository in the format
      `projects/*/locations/*/connections/*/repositories/*`.
  """
    fetchReadTokenRequest = _messages.MessageField('FetchReadTokenRequest', 1)
    repository = _messages.StringField(2, required=True)