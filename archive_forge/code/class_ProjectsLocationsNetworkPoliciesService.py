from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
class ProjectsLocationsNetworkPoliciesService(base_api.BaseApiService):
    """Service class for the projects_locations_networkPolicies resource."""
    _NAME = 'projects_locations_networkPolicies'

    def __init__(self, client):
        super(VmwareengineV1.ProjectsLocationsNetworkPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new network policy in a given VMware Engine network of a project and location (region). A new network policy cannot be created if another network policy already exists in the same scope.

      Args:
        request: (VmwareengineProjectsLocationsNetworkPoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/networkPolicies', http_method='POST', method_id='vmwareengine.projects.locations.networkPolicies.create', ordered_params=['parent'], path_params=['parent'], query_params=['networkPolicyId', 'requestId'], relative_path='v1/{+parent}/networkPolicies', request_field='networkPolicy', request_type_name='VmwareengineProjectsLocationsNetworkPoliciesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a `NetworkPolicy` resource. A network policy cannot be deleted when `NetworkService.state` is set to `RECONCILING` for either its external IP or internet access service.

      Args:
        request: (VmwareengineProjectsLocationsNetworkPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/networkPolicies/{networkPoliciesId}', http_method='DELETE', method_id='vmwareengine.projects.locations.networkPolicies.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsNetworkPoliciesDeleteRequest', response_type_name='Operation', supports_download=False)

    def FetchExternalAddresses(self, request, global_params=None):
        """Lists external IP addresses assigned to VMware workload VMs within the scope of the given network policy.

      Args:
        request: (VmwareengineProjectsLocationsNetworkPoliciesFetchExternalAddressesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FetchNetworkPolicyExternalAddressesResponse) The response message.
      """
        config = self.GetMethodConfig('FetchExternalAddresses')
        return self._RunMethod(config, request, global_params=global_params)
    FetchExternalAddresses.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/networkPolicies/{networkPoliciesId}:fetchExternalAddresses', http_method='GET', method_id='vmwareengine.projects.locations.networkPolicies.fetchExternalAddresses', ordered_params=['networkPolicy'], path_params=['networkPolicy'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+networkPolicy}:fetchExternalAddresses', request_field='', request_type_name='VmwareengineProjectsLocationsNetworkPoliciesFetchExternalAddressesRequest', response_type_name='FetchNetworkPolicyExternalAddressesResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a `NetworkPolicy` resource by its resource name.

      Args:
        request: (VmwareengineProjectsLocationsNetworkPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/networkPolicies/{networkPoliciesId}', http_method='GET', method_id='vmwareengine.projects.locations.networkPolicies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsNetworkPoliciesGetRequest', response_type_name='NetworkPolicy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists `NetworkPolicy` resources in a specified project and location.

      Args:
        request: (VmwareengineProjectsLocationsNetworkPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNetworkPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/networkPolicies', http_method='GET', method_id='vmwareengine.projects.locations.networkPolicies.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/networkPolicies', request_field='', request_type_name='VmwareengineProjectsLocationsNetworkPoliciesListRequest', response_type_name='ListNetworkPoliciesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Modifies a `NetworkPolicy` resource. Only the following fields can be updated: `internet_access`, `external_ip`, `edge_services_cidr`. Only fields specified in `updateMask` are applied. When updating a network policy, the external IP network service can only be disabled if there are no external IP addresses present in the scope of the policy. Also, a `NetworkService` cannot be updated when `NetworkService.state` is set to `RECONCILING`. During operation processing, the resource is temporarily in the `ACTIVE` state before the operation fully completes. For that period of time, you can't update the resource. Use the operation status to determine when the processing fully completes.

      Args:
        request: (VmwareengineProjectsLocationsNetworkPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/networkPolicies/{networkPoliciesId}', http_method='PATCH', method_id='vmwareengine.projects.locations.networkPolicies.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='networkPolicy', request_type_name='VmwareengineProjectsLocationsNetworkPoliciesPatchRequest', response_type_name='Operation', supports_download=False)