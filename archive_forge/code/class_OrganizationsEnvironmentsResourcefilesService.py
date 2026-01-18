from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsResourcefilesService(base_api.BaseApiService):
    """Service class for the organizations_environments_resourcefiles resource."""
    _NAME = 'organizations_environments_resourcefiles'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsResourcefilesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a resource file. Specify the `Content-Type` as `application/octet-stream` or `multipart/form-data`. For more information about resource files, see [Resource files](https://cloud.google.com/apigee/docs/api-platform/develop/resource-files).

      Args:
        request: (ApigeeOrganizationsEnvironmentsResourcefilesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ResourceFile) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/resourcefiles', http_method='POST', method_id='apigee.organizations.environments.resourcefiles.create', ordered_params=['parent'], path_params=['parent'], query_params=['name', 'type'], relative_path='v1/{+parent}/resourcefiles', request_field='googleApiHttpBody', request_type_name='ApigeeOrganizationsEnvironmentsResourcefilesCreateRequest', response_type_name='GoogleCloudApigeeV1ResourceFile', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a resource file. For more information about resource files, see [Resource files](https://cloud.google.com/apigee/docs/api-platform/develop/resource-files).

      Args:
        request: (ApigeeOrganizationsEnvironmentsResourcefilesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ResourceFile) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/resourcefiles/{type}/{name}', http_method='DELETE', method_id='apigee.organizations.environments.resourcefiles.delete', ordered_params=['parent', 'type', 'name'], path_params=['name', 'parent', 'type'], query_params=[], relative_path='v1/{+parent}/resourcefiles/{type}/{name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsResourcefilesDeleteRequest', response_type_name='GoogleCloudApigeeV1ResourceFile', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the contents of a resource file. For more information about resource files, see [Resource files](https://cloud.google.com/apigee/docs/api-platform/develop/resource-files).

      Args:
        request: (ApigeeOrganizationsEnvironmentsResourcefilesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleApiHttpBody) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/resourcefiles/{type}/{name}', http_method='GET', method_id='apigee.organizations.environments.resourcefiles.get', ordered_params=['parent', 'type', 'name'], path_params=['name', 'parent', 'type'], query_params=[], relative_path='v1/{+parent}/resourcefiles/{type}/{name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsResourcefilesGetRequest', response_type_name='GoogleApiHttpBody', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all resource files, optionally filtering by type. For more information about resource files, see [Resource files](https://cloud.google.com/apigee/docs/api-platform/develop/resource-files).

      Args:
        request: (ApigeeOrganizationsEnvironmentsResourcefilesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListEnvironmentResourcesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/resourcefiles', http_method='GET', method_id='apigee.organizations.environments.resourcefiles.list', ordered_params=['parent'], path_params=['parent'], query_params=['type'], relative_path='v1/{+parent}/resourcefiles', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsResourcefilesListRequest', response_type_name='GoogleCloudApigeeV1ListEnvironmentResourcesResponse', supports_download=False)

    def ListEnvironmentResources(self, request, global_params=None):
        """Lists all resource files, optionally filtering by type. For more information about resource files, see [Resource files](https://cloud.google.com/apigee/docs/api-platform/develop/resource-files).

      Args:
        request: (ApigeeOrganizationsEnvironmentsResourcefilesListEnvironmentResourcesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListEnvironmentResourcesResponse) The response message.
      """
        config = self.GetMethodConfig('ListEnvironmentResources')
        return self._RunMethod(config, request, global_params=global_params)
    ListEnvironmentResources.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/resourcefiles/{type}', http_method='GET', method_id='apigee.organizations.environments.resourcefiles.listEnvironmentResources', ordered_params=['parent', 'type'], path_params=['parent', 'type'], query_params=[], relative_path='v1/{+parent}/resourcefiles/{type}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsResourcefilesListEnvironmentResourcesRequest', response_type_name='GoogleCloudApigeeV1ListEnvironmentResourcesResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a resource file. Specify the `Content-Type` as `application/octet-stream` or `multipart/form-data`. For more information about resource files, see [Resource files](https://cloud.google.com/apigee/docs/api-platform/develop/resource-files).

      Args:
        request: (ApigeeOrganizationsEnvironmentsResourcefilesUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ResourceFile) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/resourcefiles/{type}/{name}', http_method='PUT', method_id='apigee.organizations.environments.resourcefiles.update', ordered_params=['parent', 'type', 'name'], path_params=['name', 'parent', 'type'], query_params=[], relative_path='v1/{+parent}/resourcefiles/{type}/{name}', request_field='googleApiHttpBody', request_type_name='ApigeeOrganizationsEnvironmentsResourcefilesUpdateRequest', response_type_name='GoogleCloudApigeeV1ResourceFile', supports_download=False)