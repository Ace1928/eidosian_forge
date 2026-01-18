from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicemanagement.v1 import servicemanagement_v1_messages as messages
class ServicesRolloutsService(base_api.BaseApiService):
    """Service class for the services_rollouts resource."""
    _NAME = 'services_rollouts'

    def __init__(self, client):
        super(ServicemanagementV1.ServicesRolloutsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new service configuration rollout. Based on rollout, the Google Service Management will roll out the service configurations to different backend services. For example, the logging configuration will be pushed to Google Cloud Logging. Please note that any previous pending and running Rollouts and associated Operations will be automatically cancelled so that the latest Rollout will not be blocked by previous Rollouts. Only the 100 most recent (in any state) and the last 10 successful (if not already part of the set of 100 most recent) rollouts are kept for each service. The rest will be deleted eventually. Operation.

      Args:
        request: (ServicemanagementServicesRolloutsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='servicemanagement.services.rollouts.create', ordered_params=['serviceName'], path_params=['serviceName'], query_params=['force'], relative_path='v1/services/{serviceName}/rollouts', request_field='rollout', request_type_name='ServicemanagementServicesRolloutsCreateRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a service configuration rollout.

      Args:
        request: (ServicemanagementServicesRolloutsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Rollout) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='servicemanagement.services.rollouts.get', ordered_params=['serviceName', 'rolloutId'], path_params=['rolloutId', 'serviceName'], query_params=[], relative_path='v1/services/{serviceName}/rollouts/{rolloutId}', request_field='', request_type_name='ServicemanagementServicesRolloutsGetRequest', response_type_name='Rollout', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the history of the service configuration rollouts for a managed service, from the newest to the oldest.

      Args:
        request: (ServicemanagementServicesRolloutsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServiceRolloutsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='servicemanagement.services.rollouts.list', ordered_params=['serviceName'], path_params=['serviceName'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/services/{serviceName}/rollouts', request_field='', request_type_name='ServicemanagementServicesRolloutsListRequest', response_type_name='ListServiceRolloutsResponse', supports_download=False)