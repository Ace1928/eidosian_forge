from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceusage.v1beta1 import serviceusage_v1beta1_messages as messages
class ServicesConsumerQuotaMetricsLimitsAdminOverridesService(base_api.BaseApiService):
    """Service class for the services_consumerQuotaMetrics_limits_adminOverrides resource."""
    _NAME = 'services_consumerQuotaMetrics_limits_adminOverrides'

    def __init__(self, client):
        super(ServiceusageV1beta1.ServicesConsumerQuotaMetricsLimitsAdminOverridesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an admin override.
An admin override is applied by an administrator of a parent folder or
parent organization of the consumer receiving the override. An admin
override is intended to limit the amount of quota the consumer can use out
of the total quota pool allocated to all children of the folder or
organization.

      Args:
        request: (ServiceusageServicesConsumerQuotaMetricsLimitsAdminOverridesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/{v1beta1Id}/{v1beta1Id1}/services/{servicesId}/consumerQuotaMetrics/{consumerQuotaMetricsId}/limits/{limitsId}/adminOverrides', http_method='POST', method_id='serviceusage.services.consumerQuotaMetrics.limits.adminOverrides.create', ordered_params=['parent'], path_params=['parent'], query_params=['force'], relative_path='v1beta1/{+parent}/adminOverrides', request_field='quotaOverride', request_type_name='ServiceusageServicesConsumerQuotaMetricsLimitsAdminOverridesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an admin override.

      Args:
        request: (ServiceusageServicesConsumerQuotaMetricsLimitsAdminOverridesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/{v1beta1Id}/{v1beta1Id1}/services/{servicesId}/consumerQuotaMetrics/{consumerQuotaMetricsId}/limits/{limitsId}/adminOverrides/{adminOverridesId}', http_method='DELETE', method_id='serviceusage.services.consumerQuotaMetrics.limits.adminOverrides.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v1beta1/{+name}', request_field='', request_type_name='ServiceusageServicesConsumerQuotaMetricsLimitsAdminOverridesDeleteRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all admin overrides on this limit.

      Args:
        request: (ServiceusageServicesConsumerQuotaMetricsLimitsAdminOverridesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAdminOverridesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/{v1beta1Id}/{v1beta1Id1}/services/{servicesId}/consumerQuotaMetrics/{consumerQuotaMetricsId}/limits/{limitsId}/adminOverrides', http_method='GET', method_id='serviceusage.services.consumerQuotaMetrics.limits.adminOverrides.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/adminOverrides', request_field='', request_type_name='ServiceusageServicesConsumerQuotaMetricsLimitsAdminOverridesListRequest', response_type_name='ListAdminOverridesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an admin override.

      Args:
        request: (ServiceusageServicesConsumerQuotaMetricsLimitsAdminOverridesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/{v1beta1Id}/{v1beta1Id1}/services/{servicesId}/consumerQuotaMetrics/{consumerQuotaMetricsId}/limits/{limitsId}/adminOverrides/{adminOverridesId}', http_method='PATCH', method_id='serviceusage.services.consumerQuotaMetrics.limits.adminOverrides.patch', ordered_params=['name'], path_params=['name'], query_params=['force', 'updateMask'], relative_path='v1beta1/{+name}', request_field='quotaOverride', request_type_name='ServiceusageServicesConsumerQuotaMetricsLimitsAdminOverridesPatchRequest', response_type_name='Operation', supports_download=False)