from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.certificatemanager.v1 import certificatemanager_v1_messages as messages
class ProjectsLocationsCertificateIssuanceConfigsService(base_api.BaseApiService):
    """Service class for the projects_locations_certificateIssuanceConfigs resource."""
    _NAME = 'projects_locations_certificateIssuanceConfigs'

    def __init__(self, client):
        super(CertificatemanagerV1.ProjectsLocationsCertificateIssuanceConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new CertificateIssuanceConfig in a given project and location.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificateIssuanceConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/certificateIssuanceConfigs', http_method='POST', method_id='certificatemanager.projects.locations.certificateIssuanceConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['certificateIssuanceConfigId'], relative_path='v1/{+parent}/certificateIssuanceConfigs', request_field='certificateIssuanceConfig', request_type_name='CertificatemanagerProjectsLocationsCertificateIssuanceConfigsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single CertificateIssuanceConfig.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificateIssuanceConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/certificateIssuanceConfigs/{certificateIssuanceConfigsId}', http_method='DELETE', method_id='certificatemanager.projects.locations.certificateIssuanceConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CertificatemanagerProjectsLocationsCertificateIssuanceConfigsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single CertificateIssuanceConfig.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificateIssuanceConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CertificateIssuanceConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/certificateIssuanceConfigs/{certificateIssuanceConfigsId}', http_method='GET', method_id='certificatemanager.projects.locations.certificateIssuanceConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CertificatemanagerProjectsLocationsCertificateIssuanceConfigsGetRequest', response_type_name='CertificateIssuanceConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Lists CertificateIssuanceConfigs in a given project and location.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificateIssuanceConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCertificateIssuanceConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/certificateIssuanceConfigs', http_method='GET', method_id='certificatemanager.projects.locations.certificateIssuanceConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/certificateIssuanceConfigs', request_field='', request_type_name='CertificatemanagerProjectsLocationsCertificateIssuanceConfigsListRequest', response_type_name='ListCertificateIssuanceConfigsResponse', supports_download=False)