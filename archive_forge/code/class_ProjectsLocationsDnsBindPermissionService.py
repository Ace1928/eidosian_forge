from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
class ProjectsLocationsDnsBindPermissionService(base_api.BaseApiService):
    """Service class for the projects_locations_dnsBindPermission resource."""
    _NAME = 'projects_locations_dnsBindPermission'

    def __init__(self, client):
        super(VmwareengineV1.ProjectsLocationsDnsBindPermissionService, self).__init__(client)
        self._upload_configs = {}

    def Grant(self, request, global_params=None):
        """Grants the bind permission to the customer provided principal(user / service account) to bind their DNS zone with the intranet VPC associated with the project. DnsBindPermission is a global resource and location can only be global.

      Args:
        request: (VmwareengineProjectsLocationsDnsBindPermissionGrantRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Grant')
        return self._RunMethod(config, request, global_params=global_params)
    Grant.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dnsBindPermission:grant', http_method='POST', method_id='vmwareengine.projects.locations.dnsBindPermission.grant', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:grant', request_field='grantDnsBindPermissionRequest', request_type_name='VmwareengineProjectsLocationsDnsBindPermissionGrantRequest', response_type_name='Operation', supports_download=False)

    def Revoke(self, request, global_params=None):
        """Revokes the bind permission from the customer provided principal(user / service account) on the intranet VPC associated with the consumer project. DnsBindPermission is a global resource and location can only be global.

      Args:
        request: (VmwareengineProjectsLocationsDnsBindPermissionRevokeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Revoke')
        return self._RunMethod(config, request, global_params=global_params)
    Revoke.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dnsBindPermission:revoke', http_method='POST', method_id='vmwareengine.projects.locations.dnsBindPermission.revoke', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:revoke', request_field='revokeDnsBindPermissionRequest', request_type_name='VmwareengineProjectsLocationsDnsBindPermissionRevokeRequest', response_type_name='Operation', supports_download=False)