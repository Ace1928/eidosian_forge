from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datastream.v1 import datastream_v1_messages as messages
class ProjectsLocationsStreamsService(base_api.BaseApiService):
    """Service class for the projects_locations_streams resource."""
    _NAME = 'projects_locations_streams'

    def __init__(self, client):
        super(DatastreamV1.ProjectsLocationsStreamsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Use this method to create a stream.

      Args:
        request: (DatastreamProjectsLocationsStreamsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streams', http_method='POST', method_id='datastream.projects.locations.streams.create', ordered_params=['parent'], path_params=['parent'], query_params=['force', 'requestId', 'streamId', 'validateOnly'], relative_path='v1/{+parent}/streams', request_field='stream', request_type_name='DatastreamProjectsLocationsStreamsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Use this method to delete a stream.

      Args:
        request: (DatastreamProjectsLocationsStreamsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streams/{streamsId}', http_method='DELETE', method_id='datastream.projects.locations.streams.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='DatastreamProjectsLocationsStreamsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Use this method to get details about a stream.

      Args:
        request: (DatastreamProjectsLocationsStreamsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Stream) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streams/{streamsId}', http_method='GET', method_id='datastream.projects.locations.streams.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DatastreamProjectsLocationsStreamsGetRequest', response_type_name='Stream', supports_download=False)

    def List(self, request, global_params=None):
        """Use this method to list streams in a project and location.

      Args:
        request: (DatastreamProjectsLocationsStreamsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListStreamsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streams', http_method='GET', method_id='datastream.projects.locations.streams.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/streams', request_field='', request_type_name='DatastreamProjectsLocationsStreamsListRequest', response_type_name='ListStreamsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Use this method to update the configuration of a stream.

      Args:
        request: (DatastreamProjectsLocationsStreamsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streams/{streamsId}', http_method='PATCH', method_id='datastream.projects.locations.streams.patch', ordered_params=['name'], path_params=['name'], query_params=['force', 'requestId', 'updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='stream', request_type_name='DatastreamProjectsLocationsStreamsPatchRequest', response_type_name='Operation', supports_download=False)

    def Run(self, request, global_params=None):
        """Use this method to start, resume or recover a stream with a non default CDC strategy. NOTE: This feature is currently experimental.

      Args:
        request: (DatastreamProjectsLocationsStreamsRunRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Run')
        return self._RunMethod(config, request, global_params=global_params)
    Run.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streams/{streamsId}:run', http_method='POST', method_id='datastream.projects.locations.streams.run', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:run', request_field='runStreamRequest', request_type_name='DatastreamProjectsLocationsStreamsRunRequest', response_type_name='Operation', supports_download=False)