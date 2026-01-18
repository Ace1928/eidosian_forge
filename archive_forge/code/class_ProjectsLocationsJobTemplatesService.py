from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.transcoder.v1 import transcoder_v1_messages as messages
class ProjectsLocationsJobTemplatesService(base_api.BaseApiService):
    """Service class for the projects_locations_jobTemplates resource."""
    _NAME = 'projects_locations_jobTemplates'

    def __init__(self, client):
        super(TranscoderV1.ProjectsLocationsJobTemplatesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a job template in the specified region.

      Args:
        request: (TranscoderProjectsLocationsJobTemplatesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (JobTemplate) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/jobTemplates', http_method='POST', method_id='transcoder.projects.locations.jobTemplates.create', ordered_params=['parent'], path_params=['parent'], query_params=['jobTemplateId'], relative_path='v1/{+parent}/jobTemplates', request_field='jobTemplate', request_type_name='TranscoderProjectsLocationsJobTemplatesCreateRequest', response_type_name='JobTemplate', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a job template.

      Args:
        request: (TranscoderProjectsLocationsJobTemplatesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/jobTemplates/{jobTemplatesId}', http_method='DELETE', method_id='transcoder.projects.locations.jobTemplates.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing'], relative_path='v1/{+name}', request_field='', request_type_name='TranscoderProjectsLocationsJobTemplatesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the job template data.

      Args:
        request: (TranscoderProjectsLocationsJobTemplatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (JobTemplate) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/jobTemplates/{jobTemplatesId}', http_method='GET', method_id='transcoder.projects.locations.jobTemplates.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='TranscoderProjectsLocationsJobTemplatesGetRequest', response_type_name='JobTemplate', supports_download=False)

    def List(self, request, global_params=None):
        """Lists job templates in the specified region.

      Args:
        request: (TranscoderProjectsLocationsJobTemplatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListJobTemplatesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/jobTemplates', http_method='GET', method_id='transcoder.projects.locations.jobTemplates.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/jobTemplates', request_field='', request_type_name='TranscoderProjectsLocationsJobTemplatesListRequest', response_type_name='ListJobTemplatesResponse', supports_download=False)