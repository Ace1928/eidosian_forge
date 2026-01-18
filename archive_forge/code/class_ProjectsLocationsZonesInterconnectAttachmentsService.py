from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.edgenetwork.v1 import edgenetwork_v1_messages as messages
class ProjectsLocationsZonesInterconnectAttachmentsService(base_api.BaseApiService):
    """Service class for the projects_locations_zones_interconnectAttachments resource."""
    _NAME = 'projects_locations_zones_interconnectAttachments'

    def __init__(self, client):
        super(EdgenetworkV1.ProjectsLocationsZonesInterconnectAttachmentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new InterconnectAttachment in a given project and location.

      Args:
        request: (EdgenetworkProjectsLocationsZonesInterconnectAttachmentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/interconnectAttachments', http_method='POST', method_id='edgenetwork.projects.locations.zones.interconnectAttachments.create', ordered_params=['parent'], path_params=['parent'], query_params=['interconnectAttachmentId', 'requestId'], relative_path='v1/{+parent}/interconnectAttachments', request_field='interconnectAttachment', request_type_name='EdgenetworkProjectsLocationsZonesInterconnectAttachmentsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single InterconnectAttachment.

      Args:
        request: (EdgenetworkProjectsLocationsZonesInterconnectAttachmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/interconnectAttachments/{interconnectAttachmentsId}', http_method='DELETE', method_id='edgenetwork.projects.locations.zones.interconnectAttachments.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='EdgenetworkProjectsLocationsZonesInterconnectAttachmentsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single InterconnectAttachment.

      Args:
        request: (EdgenetworkProjectsLocationsZonesInterconnectAttachmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InterconnectAttachment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/interconnectAttachments/{interconnectAttachmentsId}', http_method='GET', method_id='edgenetwork.projects.locations.zones.interconnectAttachments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='EdgenetworkProjectsLocationsZonesInterconnectAttachmentsGetRequest', response_type_name='InterconnectAttachment', supports_download=False)

    def List(self, request, global_params=None):
        """Lists InterconnectAttachments in a given project and location.

      Args:
        request: (EdgenetworkProjectsLocationsZonesInterconnectAttachmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListInterconnectAttachmentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/interconnectAttachments', http_method='GET', method_id='edgenetwork.projects.locations.zones.interconnectAttachments.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/interconnectAttachments', request_field='', request_type_name='EdgenetworkProjectsLocationsZonesInterconnectAttachmentsListRequest', response_type_name='ListInterconnectAttachmentsResponse', supports_download=False)