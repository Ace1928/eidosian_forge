from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceusage.v1beta1 import serviceusage_v1beta1_messages as messages
class ServicesConsumerQuotaMetricsLimitsConsumerOverridesService(base_api.BaseApiService):
    """Service class for the services_consumerQuotaMetrics_limits_consumerOverrides resource."""
    _NAME = 'services_consumerQuotaMetrics_limits_consumerOverrides'

    def __init__(self, client):
        super(ServiceusageV1beta1.ServicesConsumerQuotaMetricsLimitsConsumerOverridesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a consumer override.
A consumer override is applied to the consumer on its own authority to
limit its own quota usage. Consumer overrides cannot be used to grant more
quota than would be allowed by admin overrides, producer overrides, or the
default limit of the service.

      Args:
        request: (ServiceusageServicesConsumerQuotaMetricsLimitsConsumerOverridesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/{v1beta1Id}/{v1beta1Id1}/services/{servicesId}/consumerQuotaMetrics/{consumerQuotaMetricsId}/limits/{limitsId}/consumerOverrides', http_method='POST', method_id='serviceusage.services.consumerQuotaMetrics.limits.consumerOverrides.create', ordered_params=['parent'], path_params=['parent'], query_params=['force'], relative_path='v1beta1/{+parent}/consumerOverrides', request_field='quotaOverride', request_type_name='ServiceusageServicesConsumerQuotaMetricsLimitsConsumerOverridesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a consumer override.

      Args:
        request: (ServiceusageServicesConsumerQuotaMetricsLimitsConsumerOverridesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/{v1beta1Id}/{v1beta1Id1}/services/{servicesId}/consumerQuotaMetrics/{consumerQuotaMetricsId}/limits/{limitsId}/consumerOverrides/{consumerOverridesId}', http_method='DELETE', method_id='serviceusage.services.consumerQuotaMetrics.limits.consumerOverrides.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v1beta1/{+name}', request_field='', request_type_name='ServiceusageServicesConsumerQuotaMetricsLimitsConsumerOverridesDeleteRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all consumer overrides on this limit.

      Args:
        request: (ServiceusageServicesConsumerQuotaMetricsLimitsConsumerOverridesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConsumerOverridesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/{v1beta1Id}/{v1beta1Id1}/services/{servicesId}/consumerQuotaMetrics/{consumerQuotaMetricsId}/limits/{limitsId}/consumerOverrides', http_method='GET', method_id='serviceusage.services.consumerQuotaMetrics.limits.consumerOverrides.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/consumerOverrides', request_field='', request_type_name='ServiceusageServicesConsumerQuotaMetricsLimitsConsumerOverridesListRequest', response_type_name='ListConsumerOverridesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a consumer override.

      Args:
        request: (ServiceusageServicesConsumerQuotaMetricsLimitsConsumerOverridesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/{v1beta1Id}/{v1beta1Id1}/services/{servicesId}/consumerQuotaMetrics/{consumerQuotaMetricsId}/limits/{limitsId}/consumerOverrides/{consumerOverridesId}', http_method='PATCH', method_id='serviceusage.services.consumerQuotaMetrics.limits.consumerOverrides.patch', ordered_params=['name'], path_params=['name'], query_params=['force', 'updateMask'], relative_path='v1beta1/{+name}', request_field='quotaOverride', request_type_name='ServiceusageServicesConsumerQuotaMetricsLimitsConsumerOverridesPatchRequest', response_type_name='Operation', supports_download=False)