from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetastoreProjectsLocationsServicesRemoveIamPolicyRequest(_messages.Message):
    """A MetastoreProjectsLocationsServicesRemoveIamPolicyRequest object.

  Fields:
    removeIamPolicyRequest: A RemoveIamPolicyRequest resource to be passed as
      the request body.
    resource: Required. The relative resource name of the dataplane resource
      to remove IAM policy, in the following form:projects/{project_id}/locati
      ons/{location_id}/services/{service_id}/databases/{database_id} or proje
      cts/{project_id}/locations/{location_id}/services/{service_id}/databases
      /{database_id}/tables/{table_id}.
  """
    removeIamPolicyRequest = _messages.MessageField('RemoveIamPolicyRequest', 1)
    resource = _messages.StringField(2, required=True)