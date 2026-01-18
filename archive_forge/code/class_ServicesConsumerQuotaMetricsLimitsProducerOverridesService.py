from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceconsumermanagement.v1beta1 import serviceconsumermanagement_v1beta1_messages as messages
class ServicesConsumerQuotaMetricsLimitsProducerOverridesService(base_api.BaseApiService):
    """Service class for the services_consumerQuotaMetrics_limits_producerOverrides resource."""
    _NAME = 'services_consumerQuotaMetrics_limits_producerOverrides'

    def __init__(self, client):
        super(ServiceconsumermanagementV1beta1.ServicesConsumerQuotaMetricsLimitsProducerOverridesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a producer override.
A producer override is applied by the owner or administrator of a service
to increase or decrease the amount of quota a consumer of the service is
allowed to use.
To create multiple overrides at once, use ImportProducerOverrides instead.
If an override with the specified dimensions already exists, this call will
fail. To overwrite an existing override if one is already present ("upsert"
semantics), use ImportProducerOverrides instead.

      Args:
        request: (ServiceconsumermanagementServicesConsumerQuotaMetricsLimitsProducerOverridesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/services/{servicesId}/{servicesId1}/{servicesId2}/consumerQuotaMetrics/{consumerQuotaMetricsId}/limits/{limitsId}/producerOverrides', http_method='POST', method_id='serviceconsumermanagement.services.consumerQuotaMetrics.limits.producerOverrides.create', ordered_params=['parent'], path_params=['parent'], query_params=['force'], relative_path='v1beta1/{+parent}/producerOverrides', request_field='v1Beta1QuotaOverride', request_type_name='ServiceconsumermanagementServicesConsumerQuotaMetricsLimitsProducerOverridesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a producer override.

      Args:
        request: (ServiceconsumermanagementServicesConsumerQuotaMetricsLimitsProducerOverridesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/services/{servicesId}/{servicesId1}/{servicesId2}/consumerQuotaMetrics/{consumerQuotaMetricsId}/limits/{limitsId}/producerOverrides/{producerOverridesId}', http_method='DELETE', method_id='serviceconsumermanagement.services.consumerQuotaMetrics.limits.producerOverrides.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v1beta1/{+name}', request_field='', request_type_name='ServiceconsumermanagementServicesConsumerQuotaMetricsLimitsProducerOverridesDeleteRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all producer overrides on this limit.

      Args:
        request: (ServiceconsumermanagementServicesConsumerQuotaMetricsLimitsProducerOverridesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (V1Beta1ListProducerOverridesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/services/{servicesId}/{servicesId1}/{servicesId2}/consumerQuotaMetrics/{consumerQuotaMetricsId}/limits/{limitsId}/producerOverrides', http_method='GET', method_id='serviceconsumermanagement.services.consumerQuotaMetrics.limits.producerOverrides.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/producerOverrides', request_field='', request_type_name='ServiceconsumermanagementServicesConsumerQuotaMetricsLimitsProducerOverridesListRequest', response_type_name='V1Beta1ListProducerOverridesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a producer override.

      Args:
        request: (ServiceconsumermanagementServicesConsumerQuotaMetricsLimitsProducerOverridesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/services/{servicesId}/{servicesId1}/{servicesId2}/consumerQuotaMetrics/{consumerQuotaMetricsId}/limits/{limitsId}/producerOverrides/{producerOverridesId}', http_method='PATCH', method_id='serviceconsumermanagement.services.consumerQuotaMetrics.limits.producerOverrides.patch', ordered_params=['name'], path_params=['name'], query_params=['force', 'updateMask'], relative_path='v1beta1/{+name}', request_field='v1Beta1QuotaOverride', request_type_name='ServiceconsumermanagementServicesConsumerQuotaMetricsLimitsProducerOverridesPatchRequest', response_type_name='Operation', supports_download=False)