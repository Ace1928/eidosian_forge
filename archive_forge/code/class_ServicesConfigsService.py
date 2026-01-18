from __future__ import absolute_import
from apitools.base.py import base_api
from samples.servicemanagement_sample.servicemanagement_v1 import servicemanagement_v1_messages as messages
from the newest to the oldest.
class ServicesConfigsService(base_api.BaseApiService):
    """Service class for the services_configs resource."""
    _NAME = u'services_configs'

    def __init__(self, client):
        super(ServicemanagementV1.ServicesConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new service config (version) for a managed service. This method.
only stores the service config, but does not apply the service config to
any backend services.

      Args:
        request: (ServicemanagementServicesConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Service) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'servicemanagement.services.configs.create', ordered_params=[u'serviceName'], path_params=[u'serviceName'], query_params=[], relative_path=u'v1/services/{serviceName}/configs', request_field=u'service', request_type_name=u'ServicemanagementServicesConfigsCreateRequest', response_type_name=u'Service', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a service config (version) for a managed service. If `config_id` is.
not specified, the latest service config will be returned.

      Args:
        request: (ServicemanagementServicesConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Service) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'servicemanagement.services.configs.get', ordered_params=[u'serviceName', u'configId'], path_params=[u'configId', u'serviceName'], query_params=[], relative_path=u'v1/services/{serviceName}/configs/{configId}', request_field='', request_type_name=u'ServicemanagementServicesConfigsGetRequest', response_type_name=u'Service', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the history of the service config for a managed service,.
from the newest to the oldest.

      Args:
        request: (ServicemanagementServicesConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServiceConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'servicemanagement.services.configs.list', ordered_params=[u'serviceName'], path_params=[u'serviceName'], query_params=[u'pageSize', u'pageToken'], relative_path=u'v1/services/{serviceName}/configs', request_field='', request_type_name=u'ServicemanagementServicesConfigsListRequest', response_type_name=u'ListServiceConfigsResponse', supports_download=False)

    def Submit(self, request, global_params=None):
        """Creates a new service config (version) for a managed service based on.
user-supplied configuration sources files (for example: OpenAPI
Specification). This method stores the source configurations as well as the
generated service config. It does NOT apply the service config to any
backend services.

Operation<response: SubmitConfigSourceResponse>

      Args:
        request: (ServicemanagementServicesConfigsSubmitRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Submit')
        return self._RunMethod(config, request, global_params=global_params)
    Submit.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'servicemanagement.services.configs.submit', ordered_params=[u'serviceName'], path_params=[u'serviceName'], query_params=[], relative_path=u'v1/services/{serviceName}/configs:submit', request_field=u'submitConfigSourceRequest', request_type_name=u'ServicemanagementServicesConfigsSubmitRequest', response_type_name=u'Operation', supports_download=False)