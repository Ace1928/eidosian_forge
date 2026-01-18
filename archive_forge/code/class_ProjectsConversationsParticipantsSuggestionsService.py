from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsConversationsParticipantsSuggestionsService(base_api.BaseApiService):
    """Service class for the projects_conversations_participants_suggestions resource."""
    _NAME = 'projects_conversations_participants_suggestions'

    def __init__(self, client):
        super(DialogflowV2.ProjectsConversationsParticipantsSuggestionsService, self).__init__(client)
        self._upload_configs = {}

    def SuggestArticles(self, request, global_params=None):
        """Gets suggested articles for a participant based on specific historical messages.

      Args:
        request: (DialogflowProjectsConversationsParticipantsSuggestionsSuggestArticlesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2SuggestArticlesResponse) The response message.
      """
        config = self.GetMethodConfig('SuggestArticles')
        return self._RunMethod(config, request, global_params=global_params)
    SuggestArticles.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversations/{conversationsId}/participants/{participantsId}/suggestions:suggestArticles', http_method='POST', method_id='dialogflow.projects.conversations.participants.suggestions.suggestArticles', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/suggestions:suggestArticles', request_field='googleCloudDialogflowV2SuggestArticlesRequest', request_type_name='DialogflowProjectsConversationsParticipantsSuggestionsSuggestArticlesRequest', response_type_name='GoogleCloudDialogflowV2SuggestArticlesResponse', supports_download=False)

    def SuggestFaqAnswers(self, request, global_params=None):
        """Gets suggested faq answers for a participant based on specific historical messages.

      Args:
        request: (DialogflowProjectsConversationsParticipantsSuggestionsSuggestFaqAnswersRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2SuggestFaqAnswersResponse) The response message.
      """
        config = self.GetMethodConfig('SuggestFaqAnswers')
        return self._RunMethod(config, request, global_params=global_params)
    SuggestFaqAnswers.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversations/{conversationsId}/participants/{participantsId}/suggestions:suggestFaqAnswers', http_method='POST', method_id='dialogflow.projects.conversations.participants.suggestions.suggestFaqAnswers', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/suggestions:suggestFaqAnswers', request_field='googleCloudDialogflowV2SuggestFaqAnswersRequest', request_type_name='DialogflowProjectsConversationsParticipantsSuggestionsSuggestFaqAnswersRequest', response_type_name='GoogleCloudDialogflowV2SuggestFaqAnswersResponse', supports_download=False)

    def SuggestSmartReplies(self, request, global_params=None):
        """Gets smart replies for a participant based on specific historical messages.

      Args:
        request: (DialogflowProjectsConversationsParticipantsSuggestionsSuggestSmartRepliesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2SuggestSmartRepliesResponse) The response message.
      """
        config = self.GetMethodConfig('SuggestSmartReplies')
        return self._RunMethod(config, request, global_params=global_params)
    SuggestSmartReplies.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversations/{conversationsId}/participants/{participantsId}/suggestions:suggestSmartReplies', http_method='POST', method_id='dialogflow.projects.conversations.participants.suggestions.suggestSmartReplies', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/suggestions:suggestSmartReplies', request_field='googleCloudDialogflowV2SuggestSmartRepliesRequest', request_type_name='DialogflowProjectsConversationsParticipantsSuggestionsSuggestSmartRepliesRequest', response_type_name='GoogleCloudDialogflowV2SuggestSmartRepliesResponse', supports_download=False)