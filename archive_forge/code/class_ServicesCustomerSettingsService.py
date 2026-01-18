from __future__ import absolute_import
from apitools.base.py import base_api
from samples.servicemanagement_sample.servicemanagement_v1 import servicemanagement_v1_messages as messages
from the newest to the oldest.
class ServicesCustomerSettingsService(base_api.BaseApiService):
    """Service class for the services_customerSettings resource."""
    _NAME = u'services_customerSettings'

    def __init__(self, client):
        super(ServicemanagementV1.ServicesCustomerSettingsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieves the settings that control the specified customer's usage of the.
service.

      Args:
        request: (ServicemanagementServicesCustomerSettingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CustomerSettings) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'servicemanagement.services.customerSettings.get', ordered_params=[u'serviceName', u'customerId'], path_params=[u'customerId', u'serviceName'], query_params=[u'expand', u'view'], relative_path=u'v1/services/{serviceName}/customerSettings/{customerId}', request_field='', request_type_name=u'ServicemanagementServicesCustomerSettingsGetRequest', response_type_name=u'CustomerSettings', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates specified subset of the settings that control the specified.
customer's usage of the service.  Attempts to update a field not
controlled by the caller will result in an access denied error.

Operation<response: CustomerSettings>

      Args:
        request: (ServicemanagementServicesCustomerSettingsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'servicemanagement.services.customerSettings.patch', ordered_params=[u'serviceName', u'customerId'], path_params=[u'customerId', u'serviceName'], query_params=[u'updateMask'], relative_path=u'v1/services/{serviceName}/customerSettings/{customerId}', request_field=u'customerSettings', request_type_name=u'ServicemanagementServicesCustomerSettingsPatchRequest', response_type_name=u'Operation', supports_download=False)