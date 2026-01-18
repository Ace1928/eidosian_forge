from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
class MonitoredResourceDescriptorsService(base_api.BaseApiService):
    """Service class for the monitoredResourceDescriptors resource."""
    _NAME = 'monitoredResourceDescriptors'

    def __init__(self, client):
        super(LoggingV2.MonitoredResourceDescriptorsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists the descriptors for monitored resource types used by Logging.

      Args:
        request: (LoggingMonitoredResourceDescriptorsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMonitoredResourceDescriptorsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='logging.monitoredResourceDescriptors.list', ordered_params=[], path_params=[], query_params=['pageSize', 'pageToken'], relative_path='v2/monitoredResourceDescriptors', request_field='', request_type_name='LoggingMonitoredResourceDescriptorsListRequest', response_type_name='ListMonitoredResourceDescriptorsResponse', supports_download=False)