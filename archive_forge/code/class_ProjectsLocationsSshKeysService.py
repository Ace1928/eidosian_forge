from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
class ProjectsLocationsSshKeysService(base_api.BaseApiService):
    """Service class for the projects_locations_sshKeys resource."""
    _NAME = 'projects_locations_sshKeys'

    def __init__(self, client):
        super(BaremetalsolutionV2.ProjectsLocationsSshKeysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Register a public SSH key in the specified project for use with the interactive serial console feature.

      Args:
        request: (BaremetalsolutionProjectsLocationsSshKeysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SSHKey) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/sshKeys', http_method='POST', method_id='baremetalsolution.projects.locations.sshKeys.create', ordered_params=['parent'], path_params=['parent'], query_params=['sshKeyId'], relative_path='v2/{+parent}/sshKeys', request_field='sSHKey', request_type_name='BaremetalsolutionProjectsLocationsSshKeysCreateRequest', response_type_name='SSHKey', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a public SSH key registered in the specified project.

      Args:
        request: (BaremetalsolutionProjectsLocationsSshKeysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/sshKeys/{sshKeysId}', http_method='DELETE', method_id='baremetalsolution.projects.locations.sshKeys.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BaremetalsolutionProjectsLocationsSshKeysDeleteRequest', response_type_name='Empty', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the public SSH keys registered for the specified project. These SSH keys are used only for the interactive serial console feature.

      Args:
        request: (BaremetalsolutionProjectsLocationsSshKeysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSSHKeysResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/sshKeys', http_method='GET', method_id='baremetalsolution.projects.locations.sshKeys.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/sshKeys', request_field='', request_type_name='BaremetalsolutionProjectsLocationsSshKeysListRequest', response_type_name='ListSSHKeysResponse', supports_download=False)