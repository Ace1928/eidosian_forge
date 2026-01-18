from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsLocationsSuggestionsService(base_api.BaseApiService):
    """Service class for the projects_locations_suggestions resource."""
    _NAME = 'projects_locations_suggestions'

    def __init__(self, client):
        super(DialogflowV2.ProjectsLocationsSuggestionsService, self).__init__(client)
        self._upload_configs = {}

    def GenerateStatelessSummary(self, request, global_params=None):
        """Generates and returns a summary for a conversation that does not have a resource created for it.

      Args:
        request: (DialogflowProjectsLocationsSuggestionsGenerateStatelessSummaryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2GenerateStatelessSummaryResponse) The response message.
      """
        config = self.GetMethodConfig('GenerateStatelessSummary')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateStatelessSummary.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/suggestions:generateStatelessSummary', http_method='POST', method_id='dialogflow.projects.locations.suggestions.generateStatelessSummary', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/suggestions:generateStatelessSummary', request_field='googleCloudDialogflowV2GenerateStatelessSummaryRequest', request_type_name='DialogflowProjectsLocationsSuggestionsGenerateStatelessSummaryRequest', response_type_name='GoogleCloudDialogflowV2GenerateStatelessSummaryResponse', supports_download=False)

    def SearchKnowledge(self, request, global_params=None):
        """Get answers for the given query based on knowledge documents.

      Args:
        request: (GoogleCloudDialogflowV2SearchKnowledgeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2SearchKnowledgeResponse) The response message.
      """
        config = self.GetMethodConfig('SearchKnowledge')
        return self._RunMethod(config, request, global_params=global_params)
    SearchKnowledge.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/suggestions:searchKnowledge', http_method='POST', method_id='dialogflow.projects.locations.suggestions.searchKnowledge', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/suggestions:searchKnowledge', request_field='<request>', request_type_name='GoogleCloudDialogflowV2SearchKnowledgeRequest', response_type_name='GoogleCloudDialogflowV2SearchKnowledgeResponse', supports_download=False)