from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceusage.v2alpha import serviceusage_v2alpha_messages as messages
class CategoriesCategoryServicesService(base_api.BaseApiService):
    """Service class for the categories_categoryServices resource."""
    _NAME = 'categories_categoryServices'

    def __init__(self, client):
        super(ServiceusageV2alpha.CategoriesCategoryServicesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """List the services that belong to a given category. The supported categories are: `categories/google` and `categories/partner`.

      Args:
        request: (ServiceusageCategoriesCategoryServicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCategoryServicesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2alpha/{v2alphaId}/{v2alphaId1}/categories/{categoriesId}/categoryServices', http_method='GET', method_id='serviceusage.categories.categoryServices.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2alpha/{+parent}/categoryServices', request_field='', request_type_name='ServiceusageCategoriesCategoryServicesListRequest', response_type_name='ListCategoryServicesResponse', supports_download=False)