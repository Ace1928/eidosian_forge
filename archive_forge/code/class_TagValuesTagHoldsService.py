from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudresourcemanager.v3 import cloudresourcemanager_v3_messages as messages
class TagValuesTagHoldsService(base_api.BaseApiService):
    """Service class for the tagValues_tagHolds resource."""
    _NAME = 'tagValues_tagHolds'

    def __init__(self, client):
        super(CloudresourcemanagerV3.TagValuesTagHoldsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a TagHold. Returns ALREADY_EXISTS if a TagHold with the same resource and origin exists under the same TagValue.

      Args:
        request: (CloudresourcemanagerTagValuesTagHoldsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/tagValues/{tagValuesId}/tagHolds', http_method='POST', method_id='cloudresourcemanager.tagValues.tagHolds.create', ordered_params=['parent'], path_params=['parent'], query_params=['validateOnly'], relative_path='v3/{+parent}/tagHolds', request_field='tagHold', request_type_name='CloudresourcemanagerTagValuesTagHoldsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a TagHold.

      Args:
        request: (CloudresourcemanagerTagValuesTagHoldsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/tagValues/{tagValuesId}/tagHolds/{tagHoldsId}', http_method='DELETE', method_id='cloudresourcemanager.tagValues.tagHolds.delete', ordered_params=['name'], path_params=['name'], query_params=['validateOnly'], relative_path='v3/{+name}', request_field='', request_type_name='CloudresourcemanagerTagValuesTagHoldsDeleteRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists TagHolds under a TagValue.

      Args:
        request: (CloudresourcemanagerTagValuesTagHoldsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTagHoldsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/tagValues/{tagValuesId}/tagHolds', http_method='GET', method_id='cloudresourcemanager.tagValues.tagHolds.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v3/{+parent}/tagHolds', request_field='', request_type_name='CloudresourcemanagerTagValuesTagHoldsListRequest', response_type_name='ListTagHoldsResponse', supports_download=False)