from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1alpha3 import datacatalog_v1alpha3_messages as messages
class ProjectsTaxonomiesService(base_api.BaseApiService):
    """Service class for the projects_taxonomies resource."""
    _NAME = 'projects_taxonomies'

    def __init__(self, client):
        super(DatacatalogV1alpha3.ProjectsTaxonomiesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new taxonomy in a given project.

      Args:
        request: (DatacatalogProjectsTaxonomiesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1alpha3Taxonomy) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/taxonomies', http_method='POST', method_id='datacatalog.projects.taxonomies.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha3/{+parent}/taxonomies', request_field='googleCloudDatacatalogV1alpha3Taxonomy', request_type_name='DatacatalogProjectsTaxonomiesCreateRequest', response_type_name='GoogleCloudDatacatalogV1alpha3Taxonomy', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a taxonomy. This operation will also delete all categories in this taxonomy.

      Args:
        request: (DatacatalogProjectsTaxonomiesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/taxonomies/{taxonomiesId}', http_method='DELETE', method_id='datacatalog.projects.taxonomies.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha3/{+name}', request_field='', request_type_name='DatacatalogProjectsTaxonomiesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Export(self, request, global_params=None):
        """Exports all taxonomies and their categories in a project. This method generates SerializedTaxonomy protos with nested categories that can be used as an input for future ImportTaxonomies calls.

      Args:
        request: (DatacatalogProjectsTaxonomiesExportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1alpha3ExportTaxonomiesResponse) The response message.
      """
        config = self.GetMethodConfig('Export')
        return self._RunMethod(config, request, global_params=global_params)
    Export.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/taxonomies:export', http_method='GET', method_id='datacatalog.projects.taxonomies.export', ordered_params=['parent'], path_params=['parent'], query_params=['taxonomyNames'], relative_path='v1alpha3/{+parent}/taxonomies:export', request_field='', request_type_name='DatacatalogProjectsTaxonomiesExportRequest', response_type_name='GoogleCloudDatacatalogV1alpha3ExportTaxonomiesResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the taxonomy referred by name.

      Args:
        request: (DatacatalogProjectsTaxonomiesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1alpha3Taxonomy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/taxonomies/{taxonomiesId}', http_method='GET', method_id='datacatalog.projects.taxonomies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha3/{+name}', request_field='', request_type_name='DatacatalogProjectsTaxonomiesGetRequest', response_type_name='GoogleCloudDatacatalogV1alpha3Taxonomy', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the IAM policy for a taxonomy or a category.

      Args:
        request: (DatacatalogProjectsTaxonomiesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/taxonomies/{taxonomiesId}:getIamPolicy', http_method='POST', method_id='datacatalog.projects.taxonomies.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha3/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='DatacatalogProjectsTaxonomiesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Import(self, request, global_params=None):
        """Imports all taxonomies and their categories to a project as new taxonomies. This method provides a bulk taxonomy / category creation using nested proto structure.

      Args:
        request: (DatacatalogProjectsTaxonomiesImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1alpha3ImportTaxonomiesResponse) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/taxonomies:import', http_method='POST', method_id='datacatalog.projects.taxonomies.import', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha3/{+parent}/taxonomies:import', request_field='googleCloudDatacatalogV1alpha3ImportTaxonomiesRequest', request_type_name='DatacatalogProjectsTaxonomiesImportRequest', response_type_name='GoogleCloudDatacatalogV1alpha3ImportTaxonomiesResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all taxonomies in a project.

      Args:
        request: (DatacatalogProjectsTaxonomiesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1alpha3ListTaxonomiesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/taxonomies', http_method='GET', method_id='datacatalog.projects.taxonomies.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha3/{+parent}/taxonomies', request_field='', request_type_name='DatacatalogProjectsTaxonomiesListRequest', response_type_name='GoogleCloudDatacatalogV1alpha3ListTaxonomiesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a taxonomy.

      Args:
        request: (DatacatalogProjectsTaxonomiesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1alpha3Taxonomy) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/taxonomies/{taxonomiesId}', http_method='PATCH', method_id='datacatalog.projects.taxonomies.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha3/{+name}', request_field='googleCloudDatacatalogV1alpha3Taxonomy', request_type_name='DatacatalogProjectsTaxonomiesPatchRequest', response_type_name='GoogleCloudDatacatalogV1alpha3Taxonomy', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the IAM policy for a taxonomy or a category.

      Args:
        request: (DatacatalogProjectsTaxonomiesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/taxonomies/{taxonomiesId}:setIamPolicy', http_method='POST', method_id='datacatalog.projects.taxonomies.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha3/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='DatacatalogProjectsTaxonomiesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on specified resources.

      Args:
        request: (DatacatalogProjectsTaxonomiesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/taxonomies/{taxonomiesId}:testIamPermissions', http_method='POST', method_id='datacatalog.projects.taxonomies.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha3/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='DatacatalogProjectsTaxonomiesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)