from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsSharedflowsService(base_api.BaseApiService):
    """Service class for the organizations_sharedflows resource."""
    _NAME = 'organizations_sharedflows'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsSharedflowsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Uploads a ZIP-formatted shared flow configuration bundle to an organization. If the shared flow already exists, this creates a new revision of it. If the shared flow does not exist, this creates it. Once imported, the shared flow revision must be deployed before it can be accessed at runtime. The size limit of a shared flow bundle is 15 MB.

      Args:
        request: (ApigeeOrganizationsSharedflowsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SharedFlowRevision) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sharedflows', http_method='POST', method_id='apigee.organizations.sharedflows.create', ordered_params=['parent'], path_params=['parent'], query_params=['action', 'name'], relative_path='v1/{+parent}/sharedflows', request_field='googleApiHttpBody', request_type_name='ApigeeOrganizationsSharedflowsCreateRequest', response_type_name='GoogleCloudApigeeV1SharedFlowRevision', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a shared flow and all it's revisions. The shared flow must be undeployed before you can delete it.

      Args:
        request: (ApigeeOrganizationsSharedflowsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SharedFlow) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sharedflows/{sharedflowsId}', http_method='DELETE', method_id='apigee.organizations.sharedflows.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsSharedflowsDeleteRequest', response_type_name='GoogleCloudApigeeV1SharedFlow', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a shared flow by name, including a list of its revisions.

      Args:
        request: (ApigeeOrganizationsSharedflowsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SharedFlow) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sharedflows/{sharedflowsId}', http_method='GET', method_id='apigee.organizations.sharedflows.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsSharedflowsGetRequest', response_type_name='GoogleCloudApigeeV1SharedFlow', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all shared flows in the organization.

      Args:
        request: (ApigeeOrganizationsSharedflowsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListSharedFlowsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sharedflows', http_method='GET', method_id='apigee.organizations.sharedflows.list', ordered_params=['parent'], path_params=['parent'], query_params=['includeMetaData', 'includeRevisions'], relative_path='v1/{+parent}/sharedflows', request_field='', request_type_name='ApigeeOrganizationsSharedflowsListRequest', response_type_name='GoogleCloudApigeeV1ListSharedFlowsResponse', supports_download=False)