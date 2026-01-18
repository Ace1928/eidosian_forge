from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudcommerceconsumerprocurement.v1alpha1 import cloudcommerceconsumerprocurement_v1alpha1_messages as messages
class BillingAccountsOrdersOrderAttributionsService(base_api.BaseApiService):
    """Service class for the billingAccounts_orders_orderAttributions resource."""
    _NAME = 'billingAccounts_orders_orderAttributions'

    def __init__(self, client):
        super(CloudcommerceconsumerprocurementV1alpha1.BillingAccountsOrdersOrderAttributionsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists all OrderAttribution of the parent [Order].

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsOrdersOrderAttributionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudCommerceConsumerProcurementV1alpha1ListOrderAttributionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/billingAccounts/{billingAccountsId}/orders/{ordersId}/orderAttributions', http_method='GET', method_id='cloudcommerceconsumerprocurement.billingAccounts.orders.orderAttributions.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/orderAttributions', request_field='', request_type_name='CloudcommerceconsumerprocurementBillingAccountsOrdersOrderAttributionsListRequest', response_type_name='GoogleCloudCommerceConsumerProcurementV1alpha1ListOrderAttributionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified OrderAttribution resource.

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsOrdersOrderAttributionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/billingAccounts/{billingAccountsId}/orders/{ordersId}/orderAttributions/{orderAttributionsId}', http_method='PATCH', method_id='cloudcommerceconsumerprocurement.billingAccounts.orders.orderAttributions.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha1/{+name}', request_field='googleCloudCommerceConsumerProcurementV1alpha1OrderAttribution', request_type_name='CloudcommerceconsumerprocurementBillingAccountsOrdersOrderAttributionsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)