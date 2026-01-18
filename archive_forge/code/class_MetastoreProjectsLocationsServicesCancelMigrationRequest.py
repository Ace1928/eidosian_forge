from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetastoreProjectsLocationsServicesCancelMigrationRequest(_messages.Message):
    """A MetastoreProjectsLocationsServicesCancelMigrationRequest object.

  Fields:
    cancelMigrationRequest: A CancelMigrationRequest resource to be passed as
      the request body.
    service: Required. The relative resource name of the metastore service to
      cancel the ongoing migration to, in the following format:projects/{proje
      ct_id}/locations/{location_id}/services/{service_id}.
  """
    cancelMigrationRequest = _messages.MessageField('CancelMigrationRequest', 1)
    service = _messages.StringField(2, required=True)