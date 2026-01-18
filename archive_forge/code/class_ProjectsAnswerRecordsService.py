from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsAnswerRecordsService(base_api.BaseApiService):
    """Service class for the projects_answerRecords resource."""
    _NAME = 'projects_answerRecords'

    def __init__(self, client):
        super(DialogflowV2.ProjectsAnswerRecordsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Returns the list of all answer records in the specified project in reverse chronological order.

      Args:
        request: (DialogflowProjectsAnswerRecordsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListAnswerRecordsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/answerRecords', http_method='GET', method_id='dialogflow.projects.answerRecords.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/answerRecords', request_field='', request_type_name='DialogflowProjectsAnswerRecordsListRequest', response_type_name='GoogleCloudDialogflowV2ListAnswerRecordsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified answer record.

      Args:
        request: (DialogflowProjectsAnswerRecordsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2AnswerRecord) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/answerRecords/{answerRecordsId}', http_method='PATCH', method_id='dialogflow.projects.answerRecords.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudDialogflowV2AnswerRecord', request_type_name='DialogflowProjectsAnswerRecordsPatchRequest', response_type_name='GoogleCloudDialogflowV2AnswerRecord', supports_download=False)