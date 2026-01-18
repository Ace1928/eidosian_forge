from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sddc.v1alpha1 import sddc_v1alpha1_messages as messages
class ProjectsLocationsClusterGroupBackupsService(base_api.BaseApiService):
    """Service class for the projects_locations_clusterGroupBackups resource."""
    _NAME = 'projects_locations_clusterGroupBackups'

    def __init__(self, client):
        super(SddcV1alpha1.ProjectsLocationsClusterGroupBackupsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """`ClusterGroupBackup` is functional. A completed `longrunning.Operation` contains the new `ClusterGroupBackup` object in the response field. The returned operation is automatically deleted after a few hours, so there is no need to call `operations.delete`.

      Args:
        request: (SddcProjectsLocationsClusterGroupBackupsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroupBackups', http_method='POST', method_id='sddc.projects.locations.clusterGroupBackups.create', ordered_params=['parent'], path_params=['parent'], query_params=['clusterGroupBackupId', 'requestId'], relative_path='v1alpha1/{+parent}/clusterGroupBackups', request_field='clusterGroupBackup', request_type_name='SddcProjectsLocationsClusterGroupBackupsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a `ClusterGroupBackup`.

      Args:
        request: (SddcProjectsLocationsClusterGroupBackupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroupBackups/{clusterGroupBackupsId}', http_method='DELETE', method_id='sddc.projects.locations.clusterGroupBackups.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SddcProjectsLocationsClusterGroupBackupsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single `ClusterGroupBackup`.

      Args:
        request: (SddcProjectsLocationsClusterGroupBackupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ClusterGroupBackup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroupBackups/{clusterGroupBackupsId}', http_method='GET', method_id='sddc.projects.locations.clusterGroupBackups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SddcProjectsLocationsClusterGroupBackupsGetRequest', response_type_name='ClusterGroupBackup', supports_download=False)

    def List(self, request, global_params=None):
        """Lists `ClusterGroupBackup` objects in a given project and location (region).

      Args:
        request: (SddcProjectsLocationsClusterGroupBackupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListClusterGroupBackupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroupBackups', http_method='GET', method_id='sddc.projects.locations.clusterGroupBackups.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/clusterGroupBackups', request_field='', request_type_name='SddcProjectsLocationsClusterGroupBackupsListRequest', response_type_name='ListClusterGroupBackupsResponse', supports_download=False)