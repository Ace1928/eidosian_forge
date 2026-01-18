from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsConversationProfilesService(base_api.BaseApiService):
    """Service class for the projects_conversationProfiles resource."""
    _NAME = 'projects_conversationProfiles'

    def __init__(self, client):
        super(DialogflowV2.ProjectsConversationProfilesService, self).__init__(client)
        self._upload_configs = {}

    def ClearSuggestionFeatureConfig(self, request, global_params=None):
        """Clears a suggestion feature from a conversation profile for the given participant role. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: ClearSuggestionFeatureConfigOperationMetadata - `response`: ConversationProfile.

      Args:
        request: (DialogflowProjectsConversationProfilesClearSuggestionFeatureConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('ClearSuggestionFeatureConfig')
        return self._RunMethod(config, request, global_params=global_params)
    ClearSuggestionFeatureConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversationProfiles/{conversationProfilesId}:clearSuggestionFeatureConfig', http_method='POST', method_id='dialogflow.projects.conversationProfiles.clearSuggestionFeatureConfig', ordered_params=['conversationProfile'], path_params=['conversationProfile'], query_params=[], relative_path='v2/{+conversationProfile}:clearSuggestionFeatureConfig', request_field='googleCloudDialogflowV2ClearSuggestionFeatureConfigRequest', request_type_name='DialogflowProjectsConversationProfilesClearSuggestionFeatureConfigRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a conversation profile in the specified project. ConversationProfile.CreateTime and ConversationProfile.UpdateTime aren't populated in the response. You can retrieve them via GetConversationProfile API.

      Args:
        request: (DialogflowProjectsConversationProfilesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ConversationProfile) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversationProfiles', http_method='POST', method_id='dialogflow.projects.conversationProfiles.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/conversationProfiles', request_field='googleCloudDialogflowV2ConversationProfile', request_type_name='DialogflowProjectsConversationProfilesCreateRequest', response_type_name='GoogleCloudDialogflowV2ConversationProfile', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified conversation profile.

      Args:
        request: (DialogflowProjectsConversationProfilesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversationProfiles/{conversationProfilesId}', http_method='DELETE', method_id='dialogflow.projects.conversationProfiles.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsConversationProfilesDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the specified conversation profile.

      Args:
        request: (DialogflowProjectsConversationProfilesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ConversationProfile) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversationProfiles/{conversationProfilesId}', http_method='GET', method_id='dialogflow.projects.conversationProfiles.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsConversationProfilesGetRequest', response_type_name='GoogleCloudDialogflowV2ConversationProfile', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of all conversation profiles in the specified project.

      Args:
        request: (DialogflowProjectsConversationProfilesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListConversationProfilesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversationProfiles', http_method='GET', method_id='dialogflow.projects.conversationProfiles.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/conversationProfiles', request_field='', request_type_name='DialogflowProjectsConversationProfilesListRequest', response_type_name='GoogleCloudDialogflowV2ListConversationProfilesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified conversation profile. ConversationProfile.CreateTime and ConversationProfile.UpdateTime aren't populated in the response. You can retrieve them via GetConversationProfile API.

      Args:
        request: (DialogflowProjectsConversationProfilesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ConversationProfile) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversationProfiles/{conversationProfilesId}', http_method='PATCH', method_id='dialogflow.projects.conversationProfiles.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudDialogflowV2ConversationProfile', request_type_name='DialogflowProjectsConversationProfilesPatchRequest', response_type_name='GoogleCloudDialogflowV2ConversationProfile', supports_download=False)

    def SetSuggestionFeatureConfig(self, request, global_params=None):
        """Adds or updates a suggestion feature in a conversation profile. If the conversation profile contains the type of suggestion feature for the participant role, it will update it. Otherwise it will insert the suggestion feature. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: SetSuggestionFeatureConfigOperationMetadata - `response`: ConversationProfile If a long running operation to add or update suggestion feature config for the same conversation profile, participant role and suggestion feature type exists, please cancel the existing long running operation before sending such request, otherwise the request will be rejected.

      Args:
        request: (DialogflowProjectsConversationProfilesSetSuggestionFeatureConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('SetSuggestionFeatureConfig')
        return self._RunMethod(config, request, global_params=global_params)
    SetSuggestionFeatureConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversationProfiles/{conversationProfilesId}:setSuggestionFeatureConfig', http_method='POST', method_id='dialogflow.projects.conversationProfiles.setSuggestionFeatureConfig', ordered_params=['conversationProfile'], path_params=['conversationProfile'], query_params=[], relative_path='v2/{+conversationProfile}:setSuggestionFeatureConfig', request_field='googleCloudDialogflowV2SetSuggestionFeatureConfigRequest', request_type_name='DialogflowProjectsConversationProfilesSetSuggestionFeatureConfigRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)