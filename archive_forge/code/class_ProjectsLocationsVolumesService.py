from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
class ProjectsLocationsVolumesService(base_api.BaseApiService):
    """Service class for the projects_locations_volumes resource."""
    _NAME = 'projects_locations_volumes'

    def __init__(self, client):
        super(BaremetalsolutionV2.ProjectsLocationsVolumesService, self).__init__(client)
        self._upload_configs = {}

    def AllocateLuns(self, request, global_params=None):
        """Allocate Volume's Luns.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesAllocateLunsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AllocateLuns')
        return self._RunMethod(config, request, global_params=global_params)
    AllocateLuns.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}:allocateLuns', http_method='POST', method_id='baremetalsolution.projects.locations.volumes.allocateLuns', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}:allocateLuns', request_field='allocateLunsRequest', request_type_name='BaremetalsolutionProjectsLocationsVolumesAllocateLunsRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Create a volume.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes', http_method='POST', method_id='baremetalsolution.projects.locations.volumes.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/volumes', request_field='volume', request_type_name='BaremetalsolutionProjectsLocationsVolumesCreateRequest', response_type_name='Operation', supports_download=False)

    def CreateAndAttach(self, request, global_params=None):
        """Create a volume, allocate Luns and attach them to instances.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesCreateAndAttachRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('CreateAndAttach')
        return self._RunMethod(config, request, global_params=global_params)
    CreateAndAttach.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes:createAndAttach', http_method='POST', method_id='baremetalsolution.projects.locations.volumes.createAndAttach', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/volumes:createAndAttach', request_field='createAndAttachVolumeRequest', request_type_name='BaremetalsolutionProjectsLocationsVolumesCreateAndAttachRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete a volume. Volume shouldn't have any Luns.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}', http_method='DELETE', method_id='baremetalsolution.projects.locations.volumes.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v2/{+name}', request_field='', request_type_name='BaremetalsolutionProjectsLocationsVolumesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Evict(self, request, global_params=None):
        """Skips volume's cooloff and deletes it now. Volume must be in cooloff state.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesEvictRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Evict')
        return self._RunMethod(config, request, global_params=global_params)
    Evict.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}:evict', http_method='POST', method_id='baremetalsolution.projects.locations.volumes.evict', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:evict', request_field='evictVolumeRequest', request_type_name='BaremetalsolutionProjectsLocationsVolumesEvictRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Get details of a single storage volume.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Volume) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}', http_method='GET', method_id='baremetalsolution.projects.locations.volumes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BaremetalsolutionProjectsLocationsVolumesGetRequest', response_type_name='Volume', supports_download=False)

    def List(self, request, global_params=None):
        """List storage volumes in a given project and location.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListVolumesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes', http_method='GET', method_id='baremetalsolution.projects.locations.volumes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/volumes', request_field='', request_type_name='BaremetalsolutionProjectsLocationsVolumesListRequest', response_type_name='ListVolumesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update details of a single storage volume.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}', http_method='PATCH', method_id='baremetalsolution.projects.locations.volumes.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='volume', request_type_name='BaremetalsolutionProjectsLocationsVolumesPatchRequest', response_type_name='Operation', supports_download=False)

    def Rename(self, request, global_params=None):
        """RenameVolume sets a new name for a volume. Use with caution, previous names become immediately invalidated.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesRenameRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Volume) The response message.
      """
        config = self.GetMethodConfig('Rename')
        return self._RunMethod(config, request, global_params=global_params)
    Rename.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}:rename', http_method='POST', method_id='baremetalsolution.projects.locations.volumes.rename', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:rename', request_field='renameVolumeRequest', request_type_name='BaremetalsolutionProjectsLocationsVolumesRenameRequest', response_type_name='Volume', supports_download=False)

    def Resize(self, request, global_params=None):
        """Emergency Volume resize.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesResizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Resize')
        return self._RunMethod(config, request, global_params=global_params)
    Resize.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}:resize', http_method='POST', method_id='baremetalsolution.projects.locations.volumes.resize', ordered_params=['volume'], path_params=['volume'], query_params=[], relative_path='v2/{+volume}:resize', request_field='resizeVolumeRequest', request_type_name='BaremetalsolutionProjectsLocationsVolumesResizeRequest', response_type_name='Operation', supports_download=False)