from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1 import networksecurity_v1_messages as messages
class OrganizationsLocationsAddressGroupsService(base_api.BaseApiService):
    """Service class for the organizations_locations_addressGroups resource."""
    _NAME = 'organizations_locations_addressGroups'

    def __init__(self, client):
        super(NetworksecurityV1.OrganizationsLocationsAddressGroupsService, self).__init__(client)
        self._upload_configs = {}

    def AddItems(self, request, global_params=None):
        """Adds items to an address group.

      Args:
        request: (NetworksecurityOrganizationsLocationsAddressGroupsAddItemsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddItems')
        return self._RunMethod(config, request, global_params=global_params)
    AddItems.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/addressGroups/{addressGroupsId}:addItems', http_method='POST', method_id='networksecurity.organizations.locations.addressGroups.addItems', ordered_params=['addressGroup'], path_params=['addressGroup'], query_params=[], relative_path='v1/{+addressGroup}:addItems', request_field='addAddressGroupItemsRequest', request_type_name='NetworksecurityOrganizationsLocationsAddressGroupsAddItemsRequest', response_type_name='Operation', supports_download=False)

    def CloneItems(self, request, global_params=None):
        """Clones items from one address group to another.

      Args:
        request: (NetworksecurityOrganizationsLocationsAddressGroupsCloneItemsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('CloneItems')
        return self._RunMethod(config, request, global_params=global_params)
    CloneItems.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/addressGroups/{addressGroupsId}:cloneItems', http_method='POST', method_id='networksecurity.organizations.locations.addressGroups.cloneItems', ordered_params=['addressGroup'], path_params=['addressGroup'], query_params=[], relative_path='v1/{+addressGroup}:cloneItems', request_field='cloneAddressGroupItemsRequest', request_type_name='NetworksecurityOrganizationsLocationsAddressGroupsCloneItemsRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new address group in a given project and location.

      Args:
        request: (NetworksecurityOrganizationsLocationsAddressGroupsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/addressGroups', http_method='POST', method_id='networksecurity.organizations.locations.addressGroups.create', ordered_params=['parent'], path_params=['parent'], query_params=['addressGroupId', 'requestId'], relative_path='v1/{+parent}/addressGroups', request_field='addressGroup', request_type_name='NetworksecurityOrganizationsLocationsAddressGroupsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an address group.

      Args:
        request: (NetworksecurityOrganizationsLocationsAddressGroupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/addressGroups/{addressGroupsId}', http_method='DELETE', method_id='networksecurity.organizations.locations.addressGroups.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityOrganizationsLocationsAddressGroupsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single address group.

      Args:
        request: (NetworksecurityOrganizationsLocationsAddressGroupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AddressGroup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/addressGroups/{addressGroupsId}', http_method='GET', method_id='networksecurity.organizations.locations.addressGroups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityOrganizationsLocationsAddressGroupsGetRequest', response_type_name='AddressGroup', supports_download=False)

    def List(self, request, global_params=None):
        """Lists address groups in a given project and location.

      Args:
        request: (NetworksecurityOrganizationsLocationsAddressGroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAddressGroupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/addressGroups', http_method='GET', method_id='networksecurity.organizations.locations.addressGroups.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/addressGroups', request_field='', request_type_name='NetworksecurityOrganizationsLocationsAddressGroupsListRequest', response_type_name='ListAddressGroupsResponse', supports_download=False)

    def ListReferences(self, request, global_params=None):
        """Lists references of an address group.

      Args:
        request: (NetworksecurityOrganizationsLocationsAddressGroupsListReferencesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAddressGroupReferencesResponse) The response message.
      """
        config = self.GetMethodConfig('ListReferences')
        return self._RunMethod(config, request, global_params=global_params)
    ListReferences.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/addressGroups/{addressGroupsId}:listReferences', http_method='GET', method_id='networksecurity.organizations.locations.addressGroups.listReferences', ordered_params=['addressGroup'], path_params=['addressGroup'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+addressGroup}:listReferences', request_field='', request_type_name='NetworksecurityOrganizationsLocationsAddressGroupsListReferencesRequest', response_type_name='ListAddressGroupReferencesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates parameters of an address group.

      Args:
        request: (NetworksecurityOrganizationsLocationsAddressGroupsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/addressGroups/{addressGroupsId}', http_method='PATCH', method_id='networksecurity.organizations.locations.addressGroups.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='addressGroup', request_type_name='NetworksecurityOrganizationsLocationsAddressGroupsPatchRequest', response_type_name='Operation', supports_download=False)

    def RemoveItems(self, request, global_params=None):
        """Removes items from an address group.

      Args:
        request: (NetworksecurityOrganizationsLocationsAddressGroupsRemoveItemsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RemoveItems')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveItems.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/addressGroups/{addressGroupsId}:removeItems', http_method='POST', method_id='networksecurity.organizations.locations.addressGroups.removeItems', ordered_params=['addressGroup'], path_params=['addressGroup'], query_params=[], relative_path='v1/{+addressGroup}:removeItems', request_field='removeAddressGroupItemsRequest', request_type_name='NetworksecurityOrganizationsLocationsAddressGroupsRemoveItemsRequest', response_type_name='Operation', supports_download=False)