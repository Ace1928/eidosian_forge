from __future__ import absolute_import
from apitools.base.py import base_api
from samples.servicemanagement_sample.servicemanagement_v1 import servicemanagement_v1_messages as messages
from the newest to the oldest.
class ServicesAccessPolicyService(base_api.BaseApiService):
    """Service class for the services_accessPolicy resource."""
    _NAME = u'services_accessPolicy'

    def __init__(self, client):
        super(ServicemanagementV1.ServicesAccessPolicyService, self).__init__(client)
        self._upload_configs = {}

    def Query(self, request, global_params=None):
        """Method to query the accessibility of a service and any associated.
visibility labels for a specified user.

Members of the producer project may call this method and specify any user.

Any user may call this method, but must specify their own email address.
In this case the method will return NOT_FOUND if the user has no access to
the service.

      Args:
        request: (ServicemanagementServicesAccessPolicyQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QueryUserAccessResponse) The response message.
      """
        config = self.GetMethodConfig('Query')
        return self._RunMethod(config, request, global_params=global_params)
    Query.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'servicemanagement.services.accessPolicy.query', ordered_params=[u'serviceName'], path_params=[u'serviceName'], query_params=[u'userEmail'], relative_path=u'v1/services/{serviceName}/accessPolicy:query', request_field='', request_type_name=u'ServicemanagementServicesAccessPolicyQueryRequest', response_type_name=u'QueryUserAccessResponse', supports_download=False)