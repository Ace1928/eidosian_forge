from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsSitesApicategoriesService(base_api.BaseApiService):
    """Service class for the organizations_sites_apicategories resource."""
    _NAME = 'organizations_sites_apicategories'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsSitesApicategoriesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new API category.

      Args:
        request: (ApigeeOrganizationsSitesApicategoriesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiCategoryResponse) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sites/{sitesId}/apicategories', http_method='POST', method_id='apigee.organizations.sites.apicategories.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/apicategories', request_field='googleCloudApigeeV1ApiCategory', request_type_name='ApigeeOrganizationsSitesApicategoriesCreateRequest', response_type_name='GoogleCloudApigeeV1ApiCategoryResponse', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an API category.

      Args:
        request: (ApigeeOrganizationsSitesApicategoriesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sites/{sitesId}/apicategories/{apicategoriesId}', http_method='DELETE', method_id='apigee.organizations.sites.apicategories.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsSitesApicategoriesDeleteRequest', response_type_name='GoogleCloudApigeeV1DeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an API category.

      Args:
        request: (ApigeeOrganizationsSitesApicategoriesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiCategoryResponse) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sites/{sitesId}/apicategories/{apicategoriesId}', http_method='GET', method_id='apigee.organizations.sites.apicategories.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsSitesApicategoriesGetRequest', response_type_name='GoogleCloudApigeeV1ApiCategoryResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the API categories associated with a portal.

      Args:
        request: (ApigeeOrganizationsSitesApicategoriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListApiCategoriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sites/{sitesId}/apicategories', http_method='GET', method_id='apigee.organizations.sites.apicategories.list', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/apicategories', request_field='', request_type_name='ApigeeOrganizationsSitesApicategoriesListRequest', response_type_name='GoogleCloudApigeeV1ListApiCategoriesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an API category.

      Args:
        request: (GoogleCloudApigeeV1ApiCategory) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiCategoryResponse) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sites/{sitesId}/apicategories/{apicategoriesId}', http_method='PATCH', method_id='apigee.organizations.sites.apicategories.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='GoogleCloudApigeeV1ApiCategory', response_type_name='GoogleCloudApigeeV1ApiCategoryResponse', supports_download=False)