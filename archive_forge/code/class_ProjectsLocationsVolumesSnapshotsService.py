from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
class ProjectsLocationsVolumesSnapshotsService(base_api.BaseApiService):
    """Service class for the projects_locations_volumes_snapshots resource."""
    _NAME = 'projects_locations_volumes_snapshots'

    def __init__(self, client):
        super(BaremetalsolutionV2.ProjectsLocationsVolumesSnapshotsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Takes a snapshot of a boot volume. Returns INVALID_ARGUMENT if called for a non-boot volume.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesSnapshotsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VolumeSnapshot) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}/snapshots', http_method='POST', method_id='baremetalsolution.projects.locations.volumes.snapshots.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/snapshots', request_field='volumeSnapshot', request_type_name='BaremetalsolutionProjectsLocationsVolumesSnapshotsCreateRequest', response_type_name='VolumeSnapshot', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a volume snapshot. Returns INVALID_ARGUMENT if called for a non-boot volume.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesSnapshotsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}/snapshots/{snapshotsId}', http_method='DELETE', method_id='baremetalsolution.projects.locations.volumes.snapshots.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BaremetalsolutionProjectsLocationsVolumesSnapshotsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified snapshot resource. Returns INVALID_ARGUMENT if called for a non-boot volume.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesSnapshotsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VolumeSnapshot) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}/snapshots/{snapshotsId}', http_method='GET', method_id='baremetalsolution.projects.locations.volumes.snapshots.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BaremetalsolutionProjectsLocationsVolumesSnapshotsGetRequest', response_type_name='VolumeSnapshot', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of snapshots for the specified volume. Returns a response with an empty list of snapshots if called for a non-boot volume.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesSnapshotsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListVolumeSnapshotsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}/snapshots', http_method='GET', method_id='baremetalsolution.projects.locations.volumes.snapshots.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/snapshots', request_field='', request_type_name='BaremetalsolutionProjectsLocationsVolumesSnapshotsListRequest', response_type_name='ListVolumeSnapshotsResponse', supports_download=False)

    def RestoreVolumeSnapshot(self, request, global_params=None):
        """Uses the specified snapshot to restore its parent volume. Returns INVALID_ARGUMENT if called for a non-boot volume.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesSnapshotsRestoreVolumeSnapshotRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RestoreVolumeSnapshot')
        return self._RunMethod(config, request, global_params=global_params)
    RestoreVolumeSnapshot.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}/snapshots/{snapshotsId}:restoreVolumeSnapshot', http_method='POST', method_id='baremetalsolution.projects.locations.volumes.snapshots.restoreVolumeSnapshot', ordered_params=['volumeSnapshot'], path_params=['volumeSnapshot'], query_params=[], relative_path='v2/{+volumeSnapshot}:restoreVolumeSnapshot', request_field='restoreVolumeSnapshotRequest', request_type_name='BaremetalsolutionProjectsLocationsVolumesSnapshotsRestoreVolumeSnapshotRequest', response_type_name='Operation', supports_download=False)