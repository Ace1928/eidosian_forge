from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
class ExclusionsService(base_api.BaseApiService):
    """Service class for the exclusions resource."""
    _NAME = 'exclusions'

    def __init__(self, client):
        super(LoggingV2.ExclusionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new exclusion in the _Default sink in a specified parent resource. Only log entries belonging to that resource can be excluded. You can have up to 10 exclusions in a resource.

      Args:
        request: (LoggingExclusionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogExclusion) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/{v2Id}/{v2Id1}/exclusions', http_method='POST', method_id='logging.exclusions.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/exclusions', request_field='logExclusion', request_type_name='LoggingExclusionsCreateRequest', response_type_name='LogExclusion', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an exclusion in the _Default sink.

      Args:
        request: (LoggingExclusionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/{v2Id}/{v2Id1}/exclusions/{exclusionsId}', http_method='DELETE', method_id='logging.exclusions.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='LoggingExclusionsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the description of an exclusion in the _Default sink.

      Args:
        request: (LoggingExclusionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogExclusion) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/{v2Id}/{v2Id1}/exclusions/{exclusionsId}', http_method='GET', method_id='logging.exclusions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='LoggingExclusionsGetRequest', response_type_name='LogExclusion', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all the exclusions on the _Default sink in a parent resource.

      Args:
        request: (LoggingExclusionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListExclusionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/{v2Id}/{v2Id1}/exclusions', http_method='GET', method_id='logging.exclusions.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/exclusions', request_field='', request_type_name='LoggingExclusionsListRequest', response_type_name='ListExclusionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Changes one or more properties of an existing exclusion in the _Default sink.

      Args:
        request: (LoggingExclusionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogExclusion) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/{v2Id}/{v2Id1}/exclusions/{exclusionsId}', http_method='PATCH', method_id='logging.exclusions.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='logExclusion', request_type_name='LoggingExclusionsPatchRequest', response_type_name='LogExclusion', supports_download=False)