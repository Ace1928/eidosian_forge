from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.certificatemanager.v1alpha2 import certificatemanager_v1alpha2_messages as messages
class ProjectsLocationsCertificateMapsCertificateMapEntriesService(base_api.BaseApiService):
    """Service class for the projects_locations_certificateMaps_certificateMapEntries resource."""
    _NAME = 'projects_locations_certificateMaps_certificateMapEntries'

    def __init__(self, client):
        super(CertificatemanagerV1alpha2.ProjectsLocationsCertificateMapsCertificateMapEntriesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new CertificateMapEntry in a given project and location.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificateMapsCertificateMapEntriesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/certificateMaps/{certificateMapsId}/certificateMapEntries', http_method='POST', method_id='certificatemanager.projects.locations.certificateMaps.certificateMapEntries.create', ordered_params=['parent'], path_params=['parent'], query_params=['certificateMapEntryId'], relative_path='v1alpha2/{+parent}/certificateMapEntries', request_field='certificateMapEntry', request_type_name='CertificatemanagerProjectsLocationsCertificateMapsCertificateMapEntriesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single CertificateMapEntry.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificateMapsCertificateMapEntriesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/certificateMaps/{certificateMapsId}/certificateMapEntries/{certificateMapEntriesId}', http_method='DELETE', method_id='certificatemanager.projects.locations.certificateMaps.certificateMapEntries.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='CertificatemanagerProjectsLocationsCertificateMapsCertificateMapEntriesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single CertificateMapEntry.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificateMapsCertificateMapEntriesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CertificateMapEntry) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/certificateMaps/{certificateMapsId}/certificateMapEntries/{certificateMapEntriesId}', http_method='GET', method_id='certificatemanager.projects.locations.certificateMaps.certificateMapEntries.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='CertificatemanagerProjectsLocationsCertificateMapsCertificateMapEntriesGetRequest', response_type_name='CertificateMapEntry', supports_download=False)

    def List(self, request, global_params=None):
        """Lists CertificateMapEntries in a given project and location.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificateMapsCertificateMapEntriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCertificateMapEntriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/certificateMaps/{certificateMapsId}/certificateMapEntries', http_method='GET', method_id='certificatemanager.projects.locations.certificateMaps.certificateMapEntries.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/certificateMapEntries', request_field='', request_type_name='CertificatemanagerProjectsLocationsCertificateMapsCertificateMapEntriesListRequest', response_type_name='ListCertificateMapEntriesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a CertificateMapEntry.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificateMapsCertificateMapEntriesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/certificateMaps/{certificateMapsId}/certificateMapEntries/{certificateMapEntriesId}', http_method='PATCH', method_id='certificatemanager.projects.locations.certificateMaps.certificateMapEntries.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha2/{+name}', request_field='certificateMapEntry', request_type_name='CertificatemanagerProjectsLocationsCertificateMapsCertificateMapEntriesPatchRequest', response_type_name='Operation', supports_download=False)