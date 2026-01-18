from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetastoreProjectsLocationsServicesAlterTablePropertiesRequest(_messages.Message):
    """A MetastoreProjectsLocationsServicesAlterTablePropertiesRequest object.

  Fields:
    alterTablePropertiesRequest: A AlterTablePropertiesRequest resource to be
      passed as the request body.
    service: Required. The relative resource name of the Dataproc Metastore
      service that's being used to mutate metadata table properties, in the
      following format:projects/{project_id}/locations/{location_id}/services/
      {service_id}.
  """
    alterTablePropertiesRequest = _messages.MessageField('AlterTablePropertiesRequest', 1)
    service = _messages.StringField(2, required=True)