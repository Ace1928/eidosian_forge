from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AssuredworkloadsOrganizationsLocationsWorkloadsRestrictAllowedResourcesRequest(_messages.Message):
    """A AssuredworkloadsOrganizationsLocationsWorkloadsRestrictAllowedResource
  sRequest object.

  Fields:
    googleCloudAssuredworkloadsV1RestrictAllowedResourcesRequest: A
      GoogleCloudAssuredworkloadsV1RestrictAllowedResourcesRequest resource to
      be passed as the request body.
    name: Required. The resource name of the Workload. This is the workloads's
      relative path in the API, formatted as "organizations/{organization_id}/
      locations/{location_id}/workloads/{workload_id}". For example,
      "organizations/123/locations/us-east1/workloads/assured-workload-1".
  """
    googleCloudAssuredworkloadsV1RestrictAllowedResourcesRequest = _messages.MessageField('GoogleCloudAssuredworkloadsV1RestrictAllowedResourcesRequest', 1)
    name = _messages.StringField(2, required=True)