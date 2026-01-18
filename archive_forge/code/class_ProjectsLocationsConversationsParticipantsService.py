from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsLocationsConversationsParticipantsService(base_api.BaseApiService):
    """Service class for the projects_locations_conversations_participants resource."""
    _NAME = 'projects_locations_conversations_participants'

    def __init__(self, client):
        super(DialogflowV2.ProjectsLocationsConversationsParticipantsService, self).__init__(client)
        self._upload_configs = {}

    def AnalyzeContent(self, request, global_params=None):
        """Adds a text (chat, for example), or audio (phone recording, for example) message from a participant into the conversation. Note: Always use agent versions for production traffic sent to virtual agents. See [Versions and environments](https://cloud.google.com/dialogflow/es/docs/agents-versions).

      Args:
        request: (DialogflowProjectsLocationsConversationsParticipantsAnalyzeContentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2AnalyzeContentResponse) The response message.
      """
        config = self.GetMethodConfig('AnalyzeContent')
        return self._RunMethod(config, request, global_params=global_params)
    AnalyzeContent.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/conversations/{conversationsId}/participants/{participantsId}:analyzeContent', http_method='POST', method_id='dialogflow.projects.locations.conversations.participants.analyzeContent', ordered_params=['participant'], path_params=['participant'], query_params=[], relative_path='v2/{+participant}:analyzeContent', request_field='googleCloudDialogflowV2AnalyzeContentRequest', request_type_name='DialogflowProjectsLocationsConversationsParticipantsAnalyzeContentRequest', response_type_name='GoogleCloudDialogflowV2AnalyzeContentResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new participant in a conversation.

      Args:
        request: (DialogflowProjectsLocationsConversationsParticipantsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Participant) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/conversations/{conversationsId}/participants', http_method='POST', method_id='dialogflow.projects.locations.conversations.participants.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/participants', request_field='googleCloudDialogflowV2Participant', request_type_name='DialogflowProjectsLocationsConversationsParticipantsCreateRequest', response_type_name='GoogleCloudDialogflowV2Participant', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a conversation participant.

      Args:
        request: (DialogflowProjectsLocationsConversationsParticipantsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Participant) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/conversations/{conversationsId}/participants/{participantsId}', http_method='GET', method_id='dialogflow.projects.locations.conversations.participants.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsLocationsConversationsParticipantsGetRequest', response_type_name='GoogleCloudDialogflowV2Participant', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of all participants in the specified conversation.

      Args:
        request: (DialogflowProjectsLocationsConversationsParticipantsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListParticipantsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/conversations/{conversationsId}/participants', http_method='GET', method_id='dialogflow.projects.locations.conversations.participants.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/participants', request_field='', request_type_name='DialogflowProjectsLocationsConversationsParticipantsListRequest', response_type_name='GoogleCloudDialogflowV2ListParticipantsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified participant.

      Args:
        request: (DialogflowProjectsLocationsConversationsParticipantsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Participant) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/conversations/{conversationsId}/participants/{participantsId}', http_method='PATCH', method_id='dialogflow.projects.locations.conversations.participants.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudDialogflowV2Participant', request_type_name='DialogflowProjectsLocationsConversationsParticipantsPatchRequest', response_type_name='GoogleCloudDialogflowV2Participant', supports_download=False)