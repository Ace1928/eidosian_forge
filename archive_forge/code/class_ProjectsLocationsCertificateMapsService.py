from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.certificatemanager.v1alpha2 import certificatemanager_v1alpha2_messages as messages
class ProjectsLocationsCertificateMapsService(base_api.BaseApiService):
    """Service class for the projects_locations_certificateMaps resource."""
    _NAME = 'projects_locations_certificateMaps'

    def __init__(self, client):
        super(CertificatemanagerV1alpha2.ProjectsLocationsCertificateMapsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new CertificateMap in a given project and location.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificateMapsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/certificateMaps', http_method='POST', method_id='certificatemanager.projects.locations.certificateMaps.create', ordered_params=['parent'], path_params=['parent'], query_params=['certificateMapId'], relative_path='v1alpha2/{+parent}/certificateMaps', request_field='certificateMap', request_type_name='CertificatemanagerProjectsLocationsCertificateMapsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single CertificateMap. A Certificate Map can't be deleted if it contains Certificate Map Entries. Remove all the entries from the map before calling this method.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificateMapsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/certificateMaps/{certificateMapsId}', http_method='DELETE', method_id='certificatemanager.projects.locations.certificateMaps.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='CertificatemanagerProjectsLocationsCertificateMapsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single CertificateMap.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificateMapsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CertificateMap) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/certificateMaps/{certificateMapsId}', http_method='GET', method_id='certificatemanager.projects.locations.certificateMaps.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='CertificatemanagerProjectsLocationsCertificateMapsGetRequest', response_type_name='CertificateMap', supports_download=False)

    def List(self, request, global_params=None):
        """Lists CertificateMaps in a given project and location.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificateMapsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCertificateMapsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/certificateMaps', http_method='GET', method_id='certificatemanager.projects.locations.certificateMaps.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/certificateMaps', request_field='', request_type_name='CertificatemanagerProjectsLocationsCertificateMapsListRequest', response_type_name='ListCertificateMapsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a CertificateMap.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificateMapsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/certificateMaps/{certificateMapsId}', http_method='PATCH', method_id='certificatemanager.projects.locations.certificateMaps.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha2/{+name}', request_field='certificateMap', request_type_name='CertificatemanagerProjectsLocationsCertificateMapsPatchRequest', response_type_name='Operation', supports_download=False)