from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
class ProjectsLocationsSnapshotsService(base_api.BaseApiService):
    """Service class for the projects_locations_snapshots resource."""
    _NAME = 'projects_locations_snapshots'

    def __init__(self, client):
        super(DataflowV1b3.ProjectsLocationsSnapshotsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a snapshot.

      Args:
        request: (DataflowProjectsLocationsSnapshotsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DeleteSnapshotResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='dataflow.projects.locations.snapshots.delete', ordered_params=['projectId', 'location', 'snapshotId'], path_params=['location', 'projectId', 'snapshotId'], query_params=[], relative_path='v1b3/projects/{projectId}/locations/{location}/snapshots/{snapshotId}', request_field='', request_type_name='DataflowProjectsLocationsSnapshotsDeleteRequest', response_type_name='DeleteSnapshotResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a snapshot.

      Args:
        request: (DataflowProjectsLocationsSnapshotsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Snapshot) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dataflow.projects.locations.snapshots.get', ordered_params=['projectId', 'location', 'snapshotId'], path_params=['location', 'projectId', 'snapshotId'], query_params=[], relative_path='v1b3/projects/{projectId}/locations/{location}/snapshots/{snapshotId}', request_field='', request_type_name='DataflowProjectsLocationsSnapshotsGetRequest', response_type_name='Snapshot', supports_download=False)

    def List(self, request, global_params=None):
        """Lists snapshots.

      Args:
        request: (DataflowProjectsLocationsSnapshotsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSnapshotsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dataflow.projects.locations.snapshots.list', ordered_params=['projectId', 'location'], path_params=['location', 'projectId'], query_params=['jobId'], relative_path='v1b3/projects/{projectId}/locations/{location}/snapshots', request_field='', request_type_name='DataflowProjectsLocationsSnapshotsListRequest', response_type_name='ListSnapshotsResponse', supports_download=False)