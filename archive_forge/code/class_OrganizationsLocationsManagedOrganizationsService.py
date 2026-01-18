from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.orglifecycle.v1 import orglifecycle_v1_messages as messages
class OrganizationsLocationsManagedOrganizationsService(base_api.BaseApiService):
    """Service class for the organizations_locations_managedOrganizations resource."""
    _NAME = 'organizations_locations_managedOrganizations'

    def __init__(self, client):
        super(OrglifecycleV1.OrganizationsLocationsManagedOrganizationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new ManagedOrganization in a given organization and location.

      Args:
        request: (OrglifecycleOrganizationsLocationsManagedOrganizationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/managedOrganizations', http_method='POST', method_id='orglifecycle.organizations.locations.managedOrganizations.create', ordered_params=['parent'], path_params=['parent'], query_params=['managedOrganizationId'], relative_path='v1/{+parent}/managedOrganizations', request_field='managedOrganization', request_type_name='OrglifecycleOrganizationsLocationsManagedOrganizationsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single ManagedOrganization.

      Args:
        request: (OrglifecycleOrganizationsLocationsManagedOrganizationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/managedOrganizations/{managedOrganizationsId}', http_method='DELETE', method_id='orglifecycle.organizations.locations.managedOrganizations.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='OrglifecycleOrganizationsLocationsManagedOrganizationsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single ManagedOrganization.

      Args:
        request: (OrglifecycleOrganizationsLocationsManagedOrganizationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagedOrganization) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/managedOrganizations/{managedOrganizationsId}', http_method='GET', method_id='orglifecycle.organizations.locations.managedOrganizations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='OrglifecycleOrganizationsLocationsManagedOrganizationsGetRequest', response_type_name='ManagedOrganization', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ManagedOrganizations in a given organization and location.

      Args:
        request: (OrglifecycleOrganizationsLocationsManagedOrganizationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListManagedOrganizationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/managedOrganizations', http_method='GET', method_id='orglifecycle.organizations.locations.managedOrganizations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'showDeleted'], relative_path='v1/{+parent}/managedOrganizations', request_field='', request_type_name='OrglifecycleOrganizationsLocationsManagedOrganizationsListRequest', response_type_name='ListManagedOrganizationsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single ManagedOrganization.

      Args:
        request: (OrglifecycleOrganizationsLocationsManagedOrganizationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/managedOrganizations/{managedOrganizationsId}', http_method='PATCH', method_id='orglifecycle.organizations.locations.managedOrganizations.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='managedOrganization', request_type_name='OrglifecycleOrganizationsLocationsManagedOrganizationsPatchRequest', response_type_name='Operation', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undeletes a single ManagedOrganization, as long as it was deleted fewer than 30 days ago.

      Args:
        request: (OrglifecycleOrganizationsLocationsManagedOrganizationsUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/managedOrganizations/{managedOrganizationsId}:undelete', http_method='POST', method_id='orglifecycle.organizations.locations.managedOrganizations.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:undelete', request_field='undeleteManagedOrganizationRequest', request_type_name='OrglifecycleOrganizationsLocationsManagedOrganizationsUndeleteRequest', response_type_name='Operation', supports_download=False)