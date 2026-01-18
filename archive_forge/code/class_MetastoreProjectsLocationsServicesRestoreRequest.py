from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetastoreProjectsLocationsServicesRestoreRequest(_messages.Message):
    """A MetastoreProjectsLocationsServicesRestoreRequest object.

  Fields:
    restoreServiceRequest: A RestoreServiceRequest resource to be passed as
      the request body.
    service: Required. The relative resource name of the metastore service to
      run restore, in the following form:projects/{project_id}/locations/{loca
      tion_id}/services/{service_id}.
  """
    restoreServiceRequest = _messages.MessageField('RestoreServiceRequest', 1)
    service = _messages.StringField(2, required=True)