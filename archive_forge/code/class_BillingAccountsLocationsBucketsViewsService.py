from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
class BillingAccountsLocationsBucketsViewsService(base_api.BaseApiService):
    """Service class for the billingAccounts_locations_buckets_views resource."""
    _NAME = 'billingAccounts_locations_buckets_views'

    def __init__(self, client):
        super(LoggingV2.BillingAccountsLocationsBucketsViewsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a view over log entries in a log bucket. A bucket may contain a maximum of 30 views.

      Args:
        request: (LoggingBillingAccountsLocationsBucketsViewsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogView) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/locations/{locationsId}/buckets/{bucketsId}/views', http_method='POST', method_id='logging.billingAccounts.locations.buckets.views.create', ordered_params=['parent'], path_params=['parent'], query_params=['viewId'], relative_path='v2/{+parent}/views', request_field='logView', request_type_name='LoggingBillingAccountsLocationsBucketsViewsCreateRequest', response_type_name='LogView', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a view on a log bucket. If an UNAVAILABLE error is returned, this indicates that system is not in a state where it can delete the view. If this occurs, please try again in a few minutes.

      Args:
        request: (LoggingBillingAccountsLocationsBucketsViewsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/locations/{locationsId}/buckets/{bucketsId}/views/{viewsId}', http_method='DELETE', method_id='logging.billingAccounts.locations.buckets.views.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='LoggingBillingAccountsLocationsBucketsViewsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a view on a log bucket.

      Args:
        request: (LoggingBillingAccountsLocationsBucketsViewsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogView) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/locations/{locationsId}/buckets/{bucketsId}/views/{viewsId}', http_method='GET', method_id='logging.billingAccounts.locations.buckets.views.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='LoggingBillingAccountsLocationsBucketsViewsGetRequest', response_type_name='LogView', supports_download=False)

    def List(self, request, global_params=None):
        """Lists views on a log bucket.

      Args:
        request: (LoggingBillingAccountsLocationsBucketsViewsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListViewsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/locations/{locationsId}/buckets/{bucketsId}/views', http_method='GET', method_id='logging.billingAccounts.locations.buckets.views.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/views', request_field='', request_type_name='LoggingBillingAccountsLocationsBucketsViewsListRequest', response_type_name='ListViewsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a view on a log bucket. This method replaces the value of the filter field from the existing view with the corresponding value from the new view. If an UNAVAILABLE error is returned, this indicates that system is not in a state where it can update the view. If this occurs, please try again in a few minutes.

      Args:
        request: (LoggingBillingAccountsLocationsBucketsViewsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogView) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/locations/{locationsId}/buckets/{bucketsId}/views/{viewsId}', http_method='PATCH', method_id='logging.billingAccounts.locations.buckets.views.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='logView', request_type_name='LoggingBillingAccountsLocationsBucketsViewsPatchRequest', response_type_name='LogView', supports_download=False)