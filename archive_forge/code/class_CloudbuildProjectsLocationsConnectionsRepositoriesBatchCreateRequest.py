from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsConnectionsRepositoriesBatchCreateRequest(_messages.Message):
    """A CloudbuildProjectsLocationsConnectionsRepositoriesBatchCreateRequest
  object.

  Fields:
    batchCreateRepositoriesRequest: A BatchCreateRepositoriesRequest resource
      to be passed as the request body.
    parent: Required. The connection to contain all the repositories being
      created. Format: projects/*/locations/*/connections/* The parent field
      in the CreateRepositoryRequest messages must either be empty or match
      this field.
  """
    batchCreateRepositoriesRequest = _messages.MessageField('BatchCreateRepositoriesRequest', 1)
    parent = _messages.StringField(2, required=True)