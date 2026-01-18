from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.assuredworkloads.v1beta1 import assuredworkloads_v1beta1_messages as messages
Restrict the list of resources allowed in the Workload environment. The current list of allowed products can be found at https://cloud.google.com/assured-workloads/docs/supported-products In addition to assuredworkloads.workload.update permission, the user should also have orgpolicy.policy.set permission on the folder resource to use this functionality.

      Args:
        request: (AssuredworkloadsOrganizationsLocationsWorkloadsRestrictAllowedResourcesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAssuredworkloadsV1beta1RestrictAllowedResourcesResponse) The response message.
      