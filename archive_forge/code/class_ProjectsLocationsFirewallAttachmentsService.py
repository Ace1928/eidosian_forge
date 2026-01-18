from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1alpha1 import networksecurity_v1alpha1_messages as messages
class ProjectsLocationsFirewallAttachmentsService(base_api.BaseApiService):
    """Service class for the projects_locations_firewallAttachments resource."""
    _NAME = 'projects_locations_firewallAttachments'

    def __init__(self, client):
        super(NetworksecurityV1alpha1.ProjectsLocationsFirewallAttachmentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new FirewallAttachment in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsFirewallAttachmentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/firewallAttachments', http_method='POST', method_id='networksecurity.projects.locations.firewallAttachments.create', ordered_params=['parent'], path_params=['parent'], query_params=['firewallAttachmentId', 'requestId'], relative_path='v1alpha1/{+parent}/firewallAttachments', request_field='firewallAttachment', request_type_name='NetworksecurityProjectsLocationsFirewallAttachmentsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Attachment.

      Args:
        request: (NetworksecurityProjectsLocationsFirewallAttachmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/firewallAttachments/{firewallAttachmentsId}', http_method='DELETE', method_id='networksecurity.projects.locations.firewallAttachments.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsFirewallAttachmentsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Attachment.

      Args:
        request: (NetworksecurityProjectsLocationsFirewallAttachmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FirewallAttachment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/firewallAttachments/{firewallAttachmentsId}', http_method='GET', method_id='networksecurity.projects.locations.firewallAttachments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsFirewallAttachmentsGetRequest', response_type_name='FirewallAttachment', supports_download=False)

    def List(self, request, global_params=None):
        """Lists FirewallAttachments in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsFirewallAttachmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFirewallAttachmentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/firewallAttachments', http_method='GET', method_id='networksecurity.projects.locations.firewallAttachments.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/firewallAttachments', request_field='', request_type_name='NetworksecurityProjectsLocationsFirewallAttachmentsListRequest', response_type_name='ListFirewallAttachmentsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update a single FirewallAttachment.

      Args:
        request: (NetworksecurityProjectsLocationsFirewallAttachmentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/firewallAttachments/{firewallAttachmentsId}', http_method='PATCH', method_id='networksecurity.projects.locations.firewallAttachments.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha1/{+name}', request_field='firewallAttachment', request_type_name='NetworksecurityProjectsLocationsFirewallAttachmentsPatchRequest', response_type_name='Operation', supports_download=False)