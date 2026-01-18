from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.file.v1 import file_v1_messages as messages
class ProjectsLocationsInstancesSnapshotsService(base_api.BaseApiService):
    """Service class for the projects_locations_instances_snapshots resource."""
    _NAME = 'projects_locations_instances_snapshots'

    def __init__(self, client):
        super(FileV1.ProjectsLocationsInstancesSnapshotsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a snapshot.

      Args:
        request: (FileProjectsLocationsInstancesSnapshotsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/snapshots', http_method='POST', method_id='file.projects.locations.instances.snapshots.create', ordered_params=['parent'], path_params=['parent'], query_params=['snapshotId'], relative_path='v1/{+parent}/snapshots', request_field='snapshot', request_type_name='FileProjectsLocationsInstancesSnapshotsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a snapshot.

      Args:
        request: (FileProjectsLocationsInstancesSnapshotsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/snapshots/{snapshotsId}', http_method='DELETE', method_id='file.projects.locations.instances.snapshots.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='FileProjectsLocationsInstancesSnapshotsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the details of a specific snapshot.

      Args:
        request: (FileProjectsLocationsInstancesSnapshotsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Snapshot) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/snapshots/{snapshotsId}', http_method='GET', method_id='file.projects.locations.instances.snapshots.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='FileProjectsLocationsInstancesSnapshotsGetRequest', response_type_name='Snapshot', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all snapshots in a project for either a specified location or for all locations.

      Args:
        request: (FileProjectsLocationsInstancesSnapshotsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSnapshotsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/snapshots', http_method='GET', method_id='file.projects.locations.instances.snapshots.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/snapshots', request_field='', request_type_name='FileProjectsLocationsInstancesSnapshotsListRequest', response_type_name='ListSnapshotsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the settings of a specific snapshot.

      Args:
        request: (FileProjectsLocationsInstancesSnapshotsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/snapshots/{snapshotsId}', http_method='PATCH', method_id='file.projects.locations.instances.snapshots.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='snapshot', request_type_name='FileProjectsLocationsInstancesSnapshotsPatchRequest', response_type_name='Operation', supports_download=False)