from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.testing.v1 import testing_v1_messages as messages
class TestEnvironmentCatalogService(base_api.BaseApiService):
    """Service class for the testEnvironmentCatalog resource."""
    _NAME = 'testEnvironmentCatalog'

    def __init__(self, client):
        super(TestingV1.TestEnvironmentCatalogService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets the catalog of supported test environments. May return any of the following canonical error codes: - INVALID_ARGUMENT - if the request is malformed - NOT_FOUND - if the environment type does not exist - INTERNAL - if an internal error occurred.

      Args:
        request: (TestingTestEnvironmentCatalogGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestEnvironmentCatalog) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='testing.testEnvironmentCatalog.get', ordered_params=['environmentType'], path_params=['environmentType'], query_params=['projectId'], relative_path='v1/testEnvironmentCatalog/{environmentType}', request_field='', request_type_name='TestingTestEnvironmentCatalogGetRequest', response_type_name='TestEnvironmentCatalog', supports_download=False)