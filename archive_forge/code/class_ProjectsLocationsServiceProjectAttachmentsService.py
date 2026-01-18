from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apphub.v1alpha import apphub_v1alpha_messages as messages
class ProjectsLocationsServiceProjectAttachmentsService(base_api.BaseApiService):
    """Service class for the projects_locations_serviceProjectAttachments resource."""
    _NAME = 'projects_locations_serviceProjectAttachments'

    def __init__(self, client):
        super(ApphubV1alpha.ProjectsLocationsServiceProjectAttachmentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Attaches a service project to the host project.

      Args:
        request: (ApphubProjectsLocationsServiceProjectAttachmentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/serviceProjectAttachments', http_method='POST', method_id='apphub.projects.locations.serviceProjectAttachments.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'serviceProjectAttachmentId'], relative_path='v1alpha/{+parent}/serviceProjectAttachments', request_field='serviceProjectAttachment', request_type_name='ApphubProjectsLocationsServiceProjectAttachmentsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a service project attachment.

      Args:
        request: (ApphubProjectsLocationsServiceProjectAttachmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/serviceProjectAttachments/{serviceProjectAttachmentsId}', http_method='DELETE', method_id='apphub.projects.locations.serviceProjectAttachments.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='ApphubProjectsLocationsServiceProjectAttachmentsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a service project attachment.

      Args:
        request: (ApphubProjectsLocationsServiceProjectAttachmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceProjectAttachment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/serviceProjectAttachments/{serviceProjectAttachmentsId}', http_method='GET', method_id='apphub.projects.locations.serviceProjectAttachments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='ApphubProjectsLocationsServiceProjectAttachmentsGetRequest', response_type_name='ServiceProjectAttachment', supports_download=False)

    def List(self, request, global_params=None):
        """Lists service projects attached to the host project.

      Args:
        request: (ApphubProjectsLocationsServiceProjectAttachmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServiceProjectAttachmentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/serviceProjectAttachments', http_method='GET', method_id='apphub.projects.locations.serviceProjectAttachments.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/serviceProjectAttachments', request_field='', request_type_name='ApphubProjectsLocationsServiceProjectAttachmentsListRequest', response_type_name='ListServiceProjectAttachmentsResponse', supports_download=False)