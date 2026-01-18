from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.mediaasset.v1alpha import mediaasset_v1alpha_messages as messages
class ProjectsLocationsAssetTypesAssetsActionsService(base_api.BaseApiService):
    """Service class for the projects_locations_assetTypes_assets_actions resource."""
    _NAME = 'projects_locations_assetTypes_assets_actions'

    def __init__(self, client):
        super(MediaassetV1alpha.ProjectsLocationsAssetTypesAssetsActionsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Cancel any pending invocations under this action.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsActionsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CancelActionResponse) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}/actions/{actionsId}:cancel', http_method='POST', method_id='mediaasset.projects.locations.assetTypes.assets.actions.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}:cancel', request_field='cancelActionRequest', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsActionsCancelRequest', response_type_name='CancelActionResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single action.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsActionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Action) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}/actions/{actionsId}', http_method='GET', method_id='mediaasset.projects.locations.assetTypes.assets.actions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsActionsGetRequest', response_type_name='Action', supports_download=False)

    def List(self, request, global_params=None):
        """Lists actions in a given project and location.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsActionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListActionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}/actions', http_method='GET', method_id='mediaasset.projects.locations.assetTypes.assets.actions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/actions', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsActionsListRequest', response_type_name='ListActionsResponse', supports_download=False)

    def Trigger(self, request, global_params=None):
        """Trigger an invocation with the latest input state.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsActionsTriggerRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TriggerActionResponse) The response message.
      """
        config = self.GetMethodConfig('Trigger')
        return self._RunMethod(config, request, global_params=global_params)
    Trigger.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}/actions/{actionsId}:trigger', http_method='POST', method_id='mediaasset.projects.locations.assetTypes.assets.actions.trigger', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}:trigger', request_field='triggerActionRequest', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsActionsTriggerRequest', response_type_name='TriggerActionResponse', supports_download=False)