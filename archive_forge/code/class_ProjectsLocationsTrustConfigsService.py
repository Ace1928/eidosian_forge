from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.certificatemanager.v1 import certificatemanager_v1_messages as messages
class ProjectsLocationsTrustConfigsService(base_api.BaseApiService):
    """Service class for the projects_locations_trustConfigs resource."""
    _NAME = 'projects_locations_trustConfigs'

    def __init__(self, client):
        super(CertificatemanagerV1.ProjectsLocationsTrustConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new TrustConfig in a given project and location.

      Args:
        request: (CertificatemanagerProjectsLocationsTrustConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/trustConfigs', http_method='POST', method_id='certificatemanager.projects.locations.trustConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['trustConfigId'], relative_path='v1/{+parent}/trustConfigs', request_field='trustConfig', request_type_name='CertificatemanagerProjectsLocationsTrustConfigsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single TrustConfig.

      Args:
        request: (CertificatemanagerProjectsLocationsTrustConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/trustConfigs/{trustConfigsId}', http_method='DELETE', method_id='certificatemanager.projects.locations.trustConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v1/{+name}', request_field='', request_type_name='CertificatemanagerProjectsLocationsTrustConfigsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single TrustConfig.

      Args:
        request: (CertificatemanagerProjectsLocationsTrustConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TrustConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/trustConfigs/{trustConfigsId}', http_method='GET', method_id='certificatemanager.projects.locations.trustConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CertificatemanagerProjectsLocationsTrustConfigsGetRequest', response_type_name='TrustConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Lists TrustConfigs in a given project and location.

      Args:
        request: (CertificatemanagerProjectsLocationsTrustConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTrustConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/trustConfigs', http_method='GET', method_id='certificatemanager.projects.locations.trustConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/trustConfigs', request_field='', request_type_name='CertificatemanagerProjectsLocationsTrustConfigsListRequest', response_type_name='ListTrustConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a TrustConfig.

      Args:
        request: (CertificatemanagerProjectsLocationsTrustConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/trustConfigs/{trustConfigsId}', http_method='PATCH', method_id='certificatemanager.projects.locations.trustConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='trustConfig', request_type_name='CertificatemanagerProjectsLocationsTrustConfigsPatchRequest', response_type_name='Operation', supports_download=False)