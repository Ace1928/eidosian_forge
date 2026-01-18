from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetastoreProjectsLocationsServicesAlterLocationRequest(_messages.Message):
    """A MetastoreProjectsLocationsServicesAlterLocationRequest object.

  Fields:
    alterMetadataResourceLocationRequest: A
      AlterMetadataResourceLocationRequest resource to be passed as the
      request body.
    service: Required. The relative resource name of the metastore service to
      mutate metadata, in the following format:projects/{project_id}/locations
      /{location_id}/services/{service_id}.
  """
    alterMetadataResourceLocationRequest = _messages.MessageField('AlterMetadataResourceLocationRequest', 1)
    service = _messages.StringField(2, required=True)