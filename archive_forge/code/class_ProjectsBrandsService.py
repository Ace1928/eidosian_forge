from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iap.v1 import iap_v1_messages as messages
class ProjectsBrandsService(base_api.BaseApiService):
    """Service class for the projects_brands resource."""
    _NAME = 'projects_brands'

    def __init__(self, client):
        super(IapV1.ProjectsBrandsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Constructs a new OAuth brand for the project if one does not exist. The created brand is "internal only", meaning that OAuth clients created under it only accept requests from users who belong to the same Google Workspace organization as the project. The brand is created in an un-reviewed status. NOTE: The "internal only" status can be manually changed in the Google Cloud Console. Requires that a brand does not already exist for the project, and that the specified support email is owned by the caller.

      Args:
        request: (IapProjectsBrandsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Brand) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/brands', http_method='POST', method_id='iap.projects.brands.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/brands', request_field='brand', request_type_name='IapProjectsBrandsCreateRequest', response_type_name='Brand', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the OAuth brand of the project.

      Args:
        request: (IapProjectsBrandsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Brand) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/brands/{brandsId}', http_method='GET', method_id='iap.projects.brands.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IapProjectsBrandsGetRequest', response_type_name='Brand', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the existing brands for the project.

      Args:
        request: (IapProjectsBrandsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBrandsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/brands', http_method='GET', method_id='iap.projects.brands.list', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/brands', request_field='', request_type_name='IapProjectsBrandsListRequest', response_type_name='ListBrandsResponse', supports_download=False)