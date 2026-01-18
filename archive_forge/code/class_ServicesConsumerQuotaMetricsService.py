from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceusage.v1beta1 import serviceusage_v1beta1_messages as messages
class ServicesConsumerQuotaMetricsService(base_api.BaseApiService):
    """Service class for the services_consumerQuotaMetrics resource."""
    _NAME = 'services_consumerQuotaMetrics'

    def __init__(self, client):
        super(ServiceusageV1beta1.ServicesConsumerQuotaMetricsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieves a summary of quota information for a specific quota metric.

      Args:
        request: (ServiceusageServicesConsumerQuotaMetricsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConsumerQuotaMetric) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/{v1beta1Id}/{v1beta1Id1}/services/{servicesId}/consumerQuotaMetrics/{consumerQuotaMetricsId}', http_method='GET', method_id='serviceusage.services.consumerQuotaMetrics.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1beta1/{+name}', request_field='', request_type_name='ServiceusageServicesConsumerQuotaMetricsGetRequest', response_type_name='ConsumerQuotaMetric', supports_download=False)

    def ImportConsumerOverrides(self, request, global_params=None):
        """Create or update multiple consumer overrides atomically, all on the.
same consumer, but on many different metrics or limits.
The name field in the quota override message should not be set.

      Args:
        request: (ServiceusageServicesConsumerQuotaMetricsImportConsumerOverridesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ImportConsumerOverrides')
        return self._RunMethod(config, request, global_params=global_params)
    ImportConsumerOverrides.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/{v1beta1Id}/{v1beta1Id1}/services/{servicesId}/consumerQuotaMetrics:importConsumerOverrides', http_method='POST', method_id='serviceusage.services.consumerQuotaMetrics.importConsumerOverrides', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1beta1/{+parent}/consumerQuotaMetrics:importConsumerOverrides', request_field='importConsumerOverridesRequest', request_type_name='ServiceusageServicesConsumerQuotaMetricsImportConsumerOverridesRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a summary of all quota information visible to the service.
consumer, organized by service metric. Each metric includes information
about all of its defined limits. Each limit includes the limit
configuration (quota unit, preciseness, default value), the current
effective limit value, and all of the overrides applied to the limit.

      Args:
        request: (ServiceusageServicesConsumerQuotaMetricsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConsumerQuotaMetricsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/{v1beta1Id}/{v1beta1Id1}/services/{servicesId}/consumerQuotaMetrics', http_method='GET', method_id='serviceusage.services.consumerQuotaMetrics.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'view'], relative_path='v1beta1/{+parent}/consumerQuotaMetrics', request_field='', request_type_name='ServiceusageServicesConsumerQuotaMetricsListRequest', response_type_name='ListConsumerQuotaMetricsResponse', supports_download=False)