from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetastoreProjectsLocationsServicesCompleteMigrationRequest(_messages.Message):
    """A MetastoreProjectsLocationsServicesCompleteMigrationRequest object.

  Fields:
    completeMigrationRequest: A CompleteMigrationRequest resource to be passed
      as the request body.
    service: Required. The relative resource name of the metastore service to
      complete the migration to, in the following format:projects/{project_id}
      /locations/{location_id}/services/{service_id}.
  """
    completeMigrationRequest = _messages.MessageField('CompleteMigrationRequest', 1)
    service = _messages.StringField(2, required=True)