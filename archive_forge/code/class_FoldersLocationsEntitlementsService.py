from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.privilegedaccessmanager.v1beta import privilegedaccessmanager_v1beta_messages as messages
class FoldersLocationsEntitlementsService(base_api.BaseApiService):
    """Service class for the folders_locations_entitlements resource."""
    _NAME = 'folders_locations_entitlements'

    def __init__(self, client):
        super(PrivilegedaccessmanagerV1beta.FoldersLocationsEntitlementsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Entitlement in a given project/folder/organization and location.

      Args:
        request: (PrivilegedaccessmanagerFoldersLocationsEntitlementsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/folders/{foldersId}/locations/{locationsId}/entitlements', http_method='POST', method_id='privilegedaccessmanager.folders.locations.entitlements.create', ordered_params=['parent'], path_params=['parent'], query_params=['entitlementId', 'requestId'], relative_path='v1beta/{+parent}/entitlements', request_field='entitlement', request_type_name='PrivilegedaccessmanagerFoldersLocationsEntitlementsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Entitlement. This method can only be called when there are no in progress (ACTIVE/ACTIVATING/REVOKING) Grants under this Entitlement.

      Args:
        request: (PrivilegedaccessmanagerFoldersLocationsEntitlementsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/folders/{foldersId}/locations/{locationsId}/entitlements/{entitlementsId}', http_method='DELETE', method_id='privilegedaccessmanager.folders.locations.entitlements.delete', ordered_params=['name'], path_params=['name'], query_params=['force', 'requestId'], relative_path='v1beta/{+name}', request_field='', request_type_name='PrivilegedaccessmanagerFoldersLocationsEntitlementsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Entitlement.

      Args:
        request: (PrivilegedaccessmanagerFoldersLocationsEntitlementsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Entitlement) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/folders/{foldersId}/locations/{locationsId}/entitlements/{entitlementsId}', http_method='GET', method_id='privilegedaccessmanager.folders.locations.entitlements.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='PrivilegedaccessmanagerFoldersLocationsEntitlementsGetRequest', response_type_name='Entitlement', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Entitlements in a given project/folder/organization and location.

      Args:
        request: (PrivilegedaccessmanagerFoldersLocationsEntitlementsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListEntitlementsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/folders/{foldersId}/locations/{locationsId}/entitlements', http_method='GET', method_id='privilegedaccessmanager.folders.locations.entitlements.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/entitlements', request_field='', request_type_name='PrivilegedaccessmanagerFoldersLocationsEntitlementsListRequest', response_type_name='ListEntitlementsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the Entitlement specified in the request. The fields of the Entitlement to be updated need to be specified in the update mask. The changes made to an Entitlement are applicable only on future Grants of the Entitlement. However if new approver(s) are added or existing approver(s) are removed from the approval workflow, the changes are effective on existing grants. The following fields are currently not supported for update: * All immutable fields * Entitlement name * Resource name * Resource type * Adding an approval workflow in an entitlement which previously had no approval workflow. * Deleting the approval workflow from an entitlement. * Adding or deleting a step in the approval workflow(currently only one step is supported) Note that updates are allowed on the list of approvers in an approval workflow step.

      Args:
        request: (PrivilegedaccessmanagerFoldersLocationsEntitlementsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/folders/{foldersId}/locations/{locationsId}/entitlements/{entitlementsId}', http_method='PATCH', method_id='privilegedaccessmanager.folders.locations.entitlements.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta/{+name}', request_field='entitlement', request_type_name='PrivilegedaccessmanagerFoldersLocationsEntitlementsPatchRequest', response_type_name='Operation', supports_download=False)

    def Search(self, request, global_params=None):
        """SearchEntitlements returns Entitlements on which the caller has the specified access.

      Args:
        request: (PrivilegedaccessmanagerFoldersLocationsEntitlementsSearchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchEntitlementsResponse) The response message.
      """
        config = self.GetMethodConfig('Search')
        return self._RunMethod(config, request, global_params=global_params)
    Search.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/folders/{foldersId}/locations/{locationsId}/entitlements:search', http_method='GET', method_id='privilegedaccessmanager.folders.locations.entitlements.search', ordered_params=['parent'], path_params=['parent'], query_params=['callerAccessType', 'filter', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/entitlements:search', request_field='', request_type_name='PrivilegedaccessmanagerFoldersLocationsEntitlementsSearchRequest', response_type_name='SearchEntitlementsResponse', supports_download=False)