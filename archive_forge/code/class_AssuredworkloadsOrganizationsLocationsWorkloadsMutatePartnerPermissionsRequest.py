from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AssuredworkloadsOrganizationsLocationsWorkloadsMutatePartnerPermissionsRequest(_messages.Message):
    """A AssuredworkloadsOrganizationsLocationsWorkloadsMutatePartnerPermission
  sRequest object.

  Fields:
    googleCloudAssuredworkloadsV1MutatePartnerPermissionsRequest: A
      GoogleCloudAssuredworkloadsV1MutatePartnerPermissionsRequest resource to
      be passed as the request body.
    name: Required. The `name` field is used to identify the workload. Format:
      organizations/{org_id}/locations/{location_id}/workloads/{workload_id}
  """
    googleCloudAssuredworkloadsV1MutatePartnerPermissionsRequest = _messages.MessageField('GoogleCloudAssuredworkloadsV1MutatePartnerPermissionsRequest', 1)
    name = _messages.StringField(2, required=True)