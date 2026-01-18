from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1 import datacatalog_v1_messages as messages
class CatalogService(base_api.BaseApiService):
    """Service class for the catalog resource."""
    _NAME = 'catalog'

    def __init__(self, client):
        super(DatacatalogV1.CatalogService, self).__init__(client)
        self._upload_configs = {}

    def Search(self, request, global_params=None):
        """Searches Data Catalog for multiple resources like entries and tags that match a query. This is a [Custom Method] (https://cloud.google.com/apis/design/custom_methods) that doesn't return all information on a resource, only its ID and high level fields. To get more information, you can subsequently call specific get methods. Note: Data Catalog search queries don't guarantee full recall. Results that match your query might not be returned, even in subsequent result pages. Additionally, returned (and not returned) results can vary if you repeat search queries. For more information, see [Data Catalog search syntax] (https://cloud.google.com/data-catalog/docs/how-to/search-reference).

      Args:
        request: (GoogleCloudDatacatalogV1SearchCatalogRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1SearchCatalogResponse) The response message.
      """
        config = self.GetMethodConfig('Search')
        return self._RunMethod(config, request, global_params=global_params)
    Search.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='datacatalog.catalog.search', ordered_params=[], path_params=[], query_params=[], relative_path='v1/catalog:search', request_field='<request>', request_type_name='GoogleCloudDatacatalogV1SearchCatalogRequest', response_type_name='GoogleCloudDatacatalogV1SearchCatalogResponse', supports_download=False)