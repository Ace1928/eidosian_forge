from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
class ProjectsLocationsPrivateCloudsManagementDnsZoneBindingsService(base_api.BaseApiService):
    """Service class for the projects_locations_privateClouds_managementDnsZoneBindings resource."""
    _NAME = 'projects_locations_privateClouds_managementDnsZoneBindings'

    def __init__(self, client):
        super(VmwareengineV1.ProjectsLocationsPrivateCloudsManagementDnsZoneBindingsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new `ManagementDnsZoneBinding` resource in a private cloud. This RPC creates the DNS binding and the resource that represents the DNS binding of the consumer VPC network to the management DNS zone. A management DNS zone is the Cloud DNS cross-project binding zone that VMware Engine creates for each private cloud. It contains FQDNs and corresponding IP addresses for the private cloud's ESXi hosts and management VM appliances like vCenter and NSX Manager.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsManagementDnsZoneBindingsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/managementDnsZoneBindings', http_method='POST', method_id='vmwareengine.projects.locations.privateClouds.managementDnsZoneBindings.create', ordered_params=['parent'], path_params=['parent'], query_params=['managementDnsZoneBindingId', 'requestId'], relative_path='v1/{+parent}/managementDnsZoneBindings', request_field='managementDnsZoneBinding', request_type_name='VmwareengineProjectsLocationsPrivateCloudsManagementDnsZoneBindingsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a `ManagementDnsZoneBinding` resource. When a management DNS zone binding is deleted, the corresponding consumer VPC network is no longer bound to the management DNS zone.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsManagementDnsZoneBindingsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/managementDnsZoneBindings/{managementDnsZoneBindingsId}', http_method='DELETE', method_id='vmwareengine.projects.locations.privateClouds.managementDnsZoneBindings.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsManagementDnsZoneBindingsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a 'ManagementDnsZoneBinding' resource by its resource name.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsManagementDnsZoneBindingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagementDnsZoneBinding) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/managementDnsZoneBindings/{managementDnsZoneBindingsId}', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.managementDnsZoneBindings.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsManagementDnsZoneBindingsGetRequest', response_type_name='ManagementDnsZoneBinding', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Consumer VPCs bound to Management DNS Zone of a given private cloud.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsManagementDnsZoneBindingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListManagementDnsZoneBindingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/managementDnsZoneBindings', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.managementDnsZoneBindings.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/managementDnsZoneBindings', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsManagementDnsZoneBindingsListRequest', response_type_name='ListManagementDnsZoneBindingsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a `ManagementDnsZoneBinding` resource. Only fields specified in `update_mask` are applied.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsManagementDnsZoneBindingsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/managementDnsZoneBindings/{managementDnsZoneBindingsId}', http_method='PATCH', method_id='vmwareengine.projects.locations.privateClouds.managementDnsZoneBindings.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='managementDnsZoneBinding', request_type_name='VmwareengineProjectsLocationsPrivateCloudsManagementDnsZoneBindingsPatchRequest', response_type_name='Operation', supports_download=False)

    def Repair(self, request, global_params=None):
        """Retries to create a `ManagementDnsZoneBinding` resource that is in failed state.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsManagementDnsZoneBindingsRepairRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Repair')
        return self._RunMethod(config, request, global_params=global_params)
    Repair.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/managementDnsZoneBindings/{managementDnsZoneBindingsId}:repair', http_method='POST', method_id='vmwareengine.projects.locations.privateClouds.managementDnsZoneBindings.repair', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:repair', request_field='repairManagementDnsZoneBindingRequest', request_type_name='VmwareengineProjectsLocationsPrivateCloudsManagementDnsZoneBindingsRepairRequest', response_type_name='Operation', supports_download=False)