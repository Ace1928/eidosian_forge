from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsFlowhooksService(base_api.BaseApiService):
    """Service class for the organizations_environments_flowhooks resource."""
    _NAME = 'organizations_environments_flowhooks'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsFlowhooksService, self).__init__(client)
        self._upload_configs = {}

    def AttachSharedFlowToFlowHook(self, request, global_params=None):
        """Attaches a shared flow to a flow hook.

      Args:
        request: (ApigeeOrganizationsEnvironmentsFlowhooksAttachSharedFlowToFlowHookRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1FlowHook) The response message.
      """
        config = self.GetMethodConfig('AttachSharedFlowToFlowHook')
        return self._RunMethod(config, request, global_params=global_params)
    AttachSharedFlowToFlowHook.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/flowhooks/{flowhooksId}', http_method='PUT', method_id='apigee.organizations.environments.flowhooks.attachSharedFlowToFlowHook', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='googleCloudApigeeV1FlowHook', request_type_name='ApigeeOrganizationsEnvironmentsFlowhooksAttachSharedFlowToFlowHookRequest', response_type_name='GoogleCloudApigeeV1FlowHook', supports_download=False)

    def DetachSharedFlowFromFlowHook(self, request, global_params=None):
        """Detaches a shared flow from a flow hook.

      Args:
        request: (ApigeeOrganizationsEnvironmentsFlowhooksDetachSharedFlowFromFlowHookRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1FlowHook) The response message.
      """
        config = self.GetMethodConfig('DetachSharedFlowFromFlowHook')
        return self._RunMethod(config, request, global_params=global_params)
    DetachSharedFlowFromFlowHook.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/flowhooks/{flowhooksId}', http_method='DELETE', method_id='apigee.organizations.environments.flowhooks.detachSharedFlowFromFlowHook', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsFlowhooksDetachSharedFlowFromFlowHookRequest', response_type_name='GoogleCloudApigeeV1FlowHook', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the name of the shared flow attached to the specified flow hook. If there's no shared flow attached to the flow hook, the API does not return an error; it simply does not return a name in the response.

      Args:
        request: (ApigeeOrganizationsEnvironmentsFlowhooksGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1FlowHook) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/flowhooks/{flowhooksId}', http_method='GET', method_id='apigee.organizations.environments.flowhooks.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsFlowhooksGetRequest', response_type_name='GoogleCloudApigeeV1FlowHook', supports_download=False)