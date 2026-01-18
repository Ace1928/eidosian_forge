from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsStudiesService(base_api.BaseApiService):
    """Service class for the projects_locations_studies resource."""
    _NAME = 'projects_locations_studies'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsStudiesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Study. A resource name will be generated after creation of the Study.

      Args:
        request: (AiplatformProjectsLocationsStudiesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Study) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/studies', http_method='POST', method_id='aiplatform.projects.locations.studies.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/studies', request_field='googleCloudAiplatformV1Study', request_type_name='AiplatformProjectsLocationsStudiesCreateRequest', response_type_name='GoogleCloudAiplatformV1Study', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Study.

      Args:
        request: (AiplatformProjectsLocationsStudiesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/studies/{studiesId}', http_method='DELETE', method_id='aiplatform.projects.locations.studies.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsStudiesDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a Study by name.

      Args:
        request: (AiplatformProjectsLocationsStudiesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Study) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/studies/{studiesId}', http_method='GET', method_id='aiplatform.projects.locations.studies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsStudiesGetRequest', response_type_name='GoogleCloudAiplatformV1Study', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all the studies in a region for an associated project.

      Args:
        request: (AiplatformProjectsLocationsStudiesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListStudiesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/studies', http_method='GET', method_id='aiplatform.projects.locations.studies.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/studies', request_field='', request_type_name='AiplatformProjectsLocationsStudiesListRequest', response_type_name='GoogleCloudAiplatformV1ListStudiesResponse', supports_download=False)

    def Lookup(self, request, global_params=None):
        """Looks a study up using the user-defined display_name field instead of the fully qualified resource name.

      Args:
        request: (AiplatformProjectsLocationsStudiesLookupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Study) The response message.
      """
        config = self.GetMethodConfig('Lookup')
        return self._RunMethod(config, request, global_params=global_params)
    Lookup.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/studies:lookup', http_method='POST', method_id='aiplatform.projects.locations.studies.lookup', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/studies:lookup', request_field='googleCloudAiplatformV1LookupStudyRequest', request_type_name='AiplatformProjectsLocationsStudiesLookupRequest', response_type_name='GoogleCloudAiplatformV1Study', supports_download=False)