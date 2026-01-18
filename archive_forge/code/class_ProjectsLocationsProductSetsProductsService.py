from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vision.v1 import vision_v1_messages as messages
class ProjectsLocationsProductSetsProductsService(base_api.BaseApiService):
    """Service class for the projects_locations_productSets_products resource."""
    _NAME = 'projects_locations_productSets_products'

    def __init__(self, client):
        super(VisionV1.ProjectsLocationsProductSetsProductsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists the Products in a ProductSet, in an unspecified order. If the ProductSet does not exist, the products field of the response will be empty. Possible errors: * Returns INVALID_ARGUMENT if page_size is greater than 100 or less than 1.

      Args:
        request: (VisionProjectsLocationsProductSetsProductsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListProductsInProductSetResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/productSets/{productSetsId}/products', http_method='GET', method_id='vision.projects.locations.productSets.products.list', ordered_params=['name'], path_params=['name'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+name}/products', request_field='', request_type_name='VisionProjectsLocationsProductSetsProductsListRequest', response_type_name='ListProductsInProductSetResponse', supports_download=False)