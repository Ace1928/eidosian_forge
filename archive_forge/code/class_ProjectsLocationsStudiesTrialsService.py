from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsStudiesTrialsService(base_api.BaseApiService):
    """Service class for the projects_locations_studies_trials resource."""
    _NAME = 'projects_locations_studies_trials'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsStudiesTrialsService, self).__init__(client)
        self._upload_configs = {}

    def AddTrialMeasurement(self, request, global_params=None):
        """Adds a measurement of the objective metrics to a Trial. This measurement is assumed to have been taken before the Trial is complete.

      Args:
        request: (AiplatformProjectsLocationsStudiesTrialsAddTrialMeasurementRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Trial) The response message.
      """
        config = self.GetMethodConfig('AddTrialMeasurement')
        return self._RunMethod(config, request, global_params=global_params)
    AddTrialMeasurement.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/studies/{studiesId}/trials/{trialsId}:addTrialMeasurement', http_method='POST', method_id='aiplatform.projects.locations.studies.trials.addTrialMeasurement', ordered_params=['trialName'], path_params=['trialName'], query_params=[], relative_path='v1/{+trialName}:addTrialMeasurement', request_field='googleCloudAiplatformV1AddTrialMeasurementRequest', request_type_name='AiplatformProjectsLocationsStudiesTrialsAddTrialMeasurementRequest', response_type_name='GoogleCloudAiplatformV1Trial', supports_download=False)

    def CheckTrialEarlyStoppingState(self, request, global_params=None):
        """Checks whether a Trial should stop or not. Returns a long-running operation. When the operation is successful, it will contain a CheckTrialEarlyStoppingStateResponse.

      Args:
        request: (AiplatformProjectsLocationsStudiesTrialsCheckTrialEarlyStoppingStateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('CheckTrialEarlyStoppingState')
        return self._RunMethod(config, request, global_params=global_params)
    CheckTrialEarlyStoppingState.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/studies/{studiesId}/trials/{trialsId}:checkTrialEarlyStoppingState', http_method='POST', method_id='aiplatform.projects.locations.studies.trials.checkTrialEarlyStoppingState', ordered_params=['trialName'], path_params=['trialName'], query_params=[], relative_path='v1/{+trialName}:checkTrialEarlyStoppingState', request_field='googleCloudAiplatformV1CheckTrialEarlyStoppingStateRequest', request_type_name='AiplatformProjectsLocationsStudiesTrialsCheckTrialEarlyStoppingStateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Complete(self, request, global_params=None):
        """Marks a Trial as complete.

      Args:
        request: (AiplatformProjectsLocationsStudiesTrialsCompleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Trial) The response message.
      """
        config = self.GetMethodConfig('Complete')
        return self._RunMethod(config, request, global_params=global_params)
    Complete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/studies/{studiesId}/trials/{trialsId}:complete', http_method='POST', method_id='aiplatform.projects.locations.studies.trials.complete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:complete', request_field='googleCloudAiplatformV1CompleteTrialRequest', request_type_name='AiplatformProjectsLocationsStudiesTrialsCompleteRequest', response_type_name='GoogleCloudAiplatformV1Trial', supports_download=False)

    def Create(self, request, global_params=None):
        """Adds a user provided Trial to a Study.

      Args:
        request: (AiplatformProjectsLocationsStudiesTrialsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Trial) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/studies/{studiesId}/trials', http_method='POST', method_id='aiplatform.projects.locations.studies.trials.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/trials', request_field='googleCloudAiplatformV1Trial', request_type_name='AiplatformProjectsLocationsStudiesTrialsCreateRequest', response_type_name='GoogleCloudAiplatformV1Trial', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Trial.

      Args:
        request: (AiplatformProjectsLocationsStudiesTrialsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/studies/{studiesId}/trials/{trialsId}', http_method='DELETE', method_id='aiplatform.projects.locations.studies.trials.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsStudiesTrialsDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a Trial.

      Args:
        request: (AiplatformProjectsLocationsStudiesTrialsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Trial) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/studies/{studiesId}/trials/{trialsId}', http_method='GET', method_id='aiplatform.projects.locations.studies.trials.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsStudiesTrialsGetRequest', response_type_name='GoogleCloudAiplatformV1Trial', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the Trials associated with a Study.

      Args:
        request: (AiplatformProjectsLocationsStudiesTrialsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListTrialsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/studies/{studiesId}/trials', http_method='GET', method_id='aiplatform.projects.locations.studies.trials.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/trials', request_field='', request_type_name='AiplatformProjectsLocationsStudiesTrialsListRequest', response_type_name='GoogleCloudAiplatformV1ListTrialsResponse', supports_download=False)

    def ListOptimalTrials(self, request, global_params=None):
        """Lists the pareto-optimal Trials for multi-objective Study or the optimal Trials for single-objective Study. The definition of pareto-optimal can be checked in wiki page. https://en.wikipedia.org/wiki/Pareto_efficiency.

      Args:
        request: (AiplatformProjectsLocationsStudiesTrialsListOptimalTrialsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListOptimalTrialsResponse) The response message.
      """
        config = self.GetMethodConfig('ListOptimalTrials')
        return self._RunMethod(config, request, global_params=global_params)
    ListOptimalTrials.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/studies/{studiesId}/trials:listOptimalTrials', http_method='POST', method_id='aiplatform.projects.locations.studies.trials.listOptimalTrials', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/trials:listOptimalTrials', request_field='googleCloudAiplatformV1ListOptimalTrialsRequest', request_type_name='AiplatformProjectsLocationsStudiesTrialsListOptimalTrialsRequest', response_type_name='GoogleCloudAiplatformV1ListOptimalTrialsResponse', supports_download=False)

    def Stop(self, request, global_params=None):
        """Stops a Trial.

      Args:
        request: (AiplatformProjectsLocationsStudiesTrialsStopRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Trial) The response message.
      """
        config = self.GetMethodConfig('Stop')
        return self._RunMethod(config, request, global_params=global_params)
    Stop.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/studies/{studiesId}/trials/{trialsId}:stop', http_method='POST', method_id='aiplatform.projects.locations.studies.trials.stop', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:stop', request_field='googleCloudAiplatformV1StopTrialRequest', request_type_name='AiplatformProjectsLocationsStudiesTrialsStopRequest', response_type_name='GoogleCloudAiplatformV1Trial', supports_download=False)

    def Suggest(self, request, global_params=None):
        """Adds one or more Trials to a Study, with parameter values suggested by Vertex AI Vizier. Returns a long-running operation associated with the generation of Trial suggestions. When this long-running operation succeeds, it will contain a SuggestTrialsResponse.

      Args:
        request: (AiplatformProjectsLocationsStudiesTrialsSuggestRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Suggest')
        return self._RunMethod(config, request, global_params=global_params)
    Suggest.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/studies/{studiesId}/trials:suggest', http_method='POST', method_id='aiplatform.projects.locations.studies.trials.suggest', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/trials:suggest', request_field='googleCloudAiplatformV1SuggestTrialsRequest', request_type_name='AiplatformProjectsLocationsStudiesTrialsSuggestRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)