from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudcommerceconsumerprocurement.v1alpha1 import cloudcommerceconsumerprocurement_v1alpha1_messages as messages
class BillingAccountsOrdersEventsService(base_api.BaseApiService):
    """Service class for the billingAccounts_orders_events resource."""
    _NAME = 'billingAccounts_orders_events'

    def __init__(self, client):
        super(CloudcommerceconsumerprocurementV1alpha1.BillingAccountsOrdersEventsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Returns the list of events associated with an order.

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsOrdersEventsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudCommerceConsumerProcurementV1alpha1ListEventsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/billingAccounts/{billingAccountsId}/orders/{ordersId}/events', http_method='GET', method_id='cloudcommerceconsumerprocurement.billingAccounts.orders.events.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/events', request_field='', request_type_name='CloudcommerceconsumerprocurementBillingAccountsOrdersEventsListRequest', response_type_name='GoogleCloudCommerceConsumerProcurementV1alpha1ListEventsResponse', supports_download=False)