from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1alpha3 import datacatalog_v1alpha3_messages as messages
class ProjectsTaxonomiesCategoriesService(base_api.BaseApiService):
    """Service class for the projects_taxonomies_categories resource."""
    _NAME = 'projects_taxonomies_categories'

    def __init__(self, client):
        super(DatacatalogV1alpha3.ProjectsTaxonomiesCategoriesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a category in a taxonomy.

      Args:
        request: (DatacatalogProjectsTaxonomiesCategoriesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1alpha3Category) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/taxonomies/{taxonomiesId}/categories', http_method='POST', method_id='datacatalog.projects.taxonomies.categories.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha3/{+parent}/categories', request_field='googleCloudDatacatalogV1alpha3Category', request_type_name='DatacatalogProjectsTaxonomiesCategoriesCreateRequest', response_type_name='GoogleCloudDatacatalogV1alpha3Category', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a category. Also deletes all of its descendant categories.

      Args:
        request: (DatacatalogProjectsTaxonomiesCategoriesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/taxonomies/{taxonomiesId}/categories/{categoriesId}', http_method='DELETE', method_id='datacatalog.projects.taxonomies.categories.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha3/{+name}', request_field='', request_type_name='DatacatalogProjectsTaxonomiesCategoriesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a category.

      Args:
        request: (DatacatalogProjectsTaxonomiesCategoriesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1alpha3Category) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/taxonomies/{taxonomiesId}/categories/{categoriesId}', http_method='GET', method_id='datacatalog.projects.taxonomies.categories.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha3/{+name}', request_field='', request_type_name='DatacatalogProjectsTaxonomiesCategoriesGetRequest', response_type_name='GoogleCloudDatacatalogV1alpha3Category', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the IAM policy for a taxonomy or a category.

      Args:
        request: (DatacatalogProjectsTaxonomiesCategoriesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/taxonomies/{taxonomiesId}/categories/{categoriesId}:getIamPolicy', http_method='POST', method_id='datacatalog.projects.taxonomies.categories.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha3/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='DatacatalogProjectsTaxonomiesCategoriesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all categories in a taxonomy.

      Args:
        request: (DatacatalogProjectsTaxonomiesCategoriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1alpha3ListCategoriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/taxonomies/{taxonomiesId}/categories', http_method='GET', method_id='datacatalog.projects.taxonomies.categories.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha3/{+parent}/categories', request_field='', request_type_name='DatacatalogProjectsTaxonomiesCategoriesListRequest', response_type_name='GoogleCloudDatacatalogV1alpha3ListCategoriesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a category.

      Args:
        request: (DatacatalogProjectsTaxonomiesCategoriesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1alpha3Category) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/taxonomies/{taxonomiesId}/categories/{categoriesId}', http_method='PATCH', method_id='datacatalog.projects.taxonomies.categories.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha3/{+name}', request_field='googleCloudDatacatalogV1alpha3Category', request_type_name='DatacatalogProjectsTaxonomiesCategoriesPatchRequest', response_type_name='GoogleCloudDatacatalogV1alpha3Category', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the IAM policy for a taxonomy or a category.

      Args:
        request: (DatacatalogProjectsTaxonomiesCategoriesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/taxonomies/{taxonomiesId}/categories/{categoriesId}:setIamPolicy', http_method='POST', method_id='datacatalog.projects.taxonomies.categories.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha3/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='DatacatalogProjectsTaxonomiesCategoriesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on specified resources.

      Args:
        request: (DatacatalogProjectsTaxonomiesCategoriesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/taxonomies/{taxonomiesId}/categories/{categoriesId}:testIamPermissions', http_method='POST', method_id='datacatalog.projects.taxonomies.categories.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha3/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='DatacatalogProjectsTaxonomiesCategoriesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)