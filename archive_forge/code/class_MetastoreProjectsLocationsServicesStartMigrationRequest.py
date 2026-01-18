from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetastoreProjectsLocationsServicesStartMigrationRequest(_messages.Message):
    """A MetastoreProjectsLocationsServicesStartMigrationRequest object.

  Fields:
    service: Required. The relative resource name of the metastore service to
      start migrating to, in the following format:projects/{project_id}/locati
      ons/{location_id}/services/{service_id}.
    startMigrationRequest: A StartMigrationRequest resource to be passed as
      the request body.
  """
    service = _messages.StringField(1, required=True)
    startMigrationRequest = _messages.MessageField('StartMigrationRequest', 2)