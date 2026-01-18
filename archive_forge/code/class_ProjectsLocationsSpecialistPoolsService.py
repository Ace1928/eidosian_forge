from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsSpecialistPoolsService(base_api.BaseApiService):
    """Service class for the projects_locations_specialistPools resource."""
    _NAME = 'projects_locations_specialistPools'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsSpecialistPoolsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a SpecialistPool.

      Args:
        request: (AiplatformProjectsLocationsSpecialistPoolsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/specialistPools', http_method='POST', method_id='aiplatform.projects.locations.specialistPools.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/specialistPools', request_field='googleCloudAiplatformV1SpecialistPool', request_type_name='AiplatformProjectsLocationsSpecialistPoolsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a SpecialistPool as well as all Specialists in the pool.

      Args:
        request: (AiplatformProjectsLocationsSpecialistPoolsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/specialistPools/{specialistPoolsId}', http_method='DELETE', method_id='aiplatform.projects.locations.specialistPools.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsSpecialistPoolsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a SpecialistPool.

      Args:
        request: (AiplatformProjectsLocationsSpecialistPoolsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1SpecialistPool) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/specialistPools/{specialistPoolsId}', http_method='GET', method_id='aiplatform.projects.locations.specialistPools.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsSpecialistPoolsGetRequest', response_type_name='GoogleCloudAiplatformV1SpecialistPool', supports_download=False)

    def List(self, request, global_params=None):
        """Lists SpecialistPools in a Location.

      Args:
        request: (AiplatformProjectsLocationsSpecialistPoolsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListSpecialistPoolsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/specialistPools', http_method='GET', method_id='aiplatform.projects.locations.specialistPools.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/specialistPools', request_field='', request_type_name='AiplatformProjectsLocationsSpecialistPoolsListRequest', response_type_name='GoogleCloudAiplatformV1ListSpecialistPoolsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a SpecialistPool.

      Args:
        request: (AiplatformProjectsLocationsSpecialistPoolsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/specialistPools/{specialistPoolsId}', http_method='PATCH', method_id='aiplatform.projects.locations.specialistPools.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1SpecialistPool', request_type_name='AiplatformProjectsLocationsSpecialistPoolsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)