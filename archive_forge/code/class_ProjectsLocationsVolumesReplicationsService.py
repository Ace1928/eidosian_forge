from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.netapp.v1 import netapp_v1_messages as messages
class ProjectsLocationsVolumesReplicationsService(base_api.BaseApiService):
    """Service class for the projects_locations_volumes_replications resource."""
    _NAME = 'projects_locations_volumes_replications'

    def __init__(self, client):
        super(NetappV1.ProjectsLocationsVolumesReplicationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a new replication for a volume.

      Args:
        request: (NetappProjectsLocationsVolumesReplicationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}/replications', http_method='POST', method_id='netapp.projects.locations.volumes.replications.create', ordered_params=['parent'], path_params=['parent'], query_params=['replicationId'], relative_path='v1/{+parent}/replications', request_field='replication', request_type_name='NetappProjectsLocationsVolumesReplicationsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a replication.

      Args:
        request: (NetappProjectsLocationsVolumesReplicationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}/replications/{replicationsId}', http_method='DELETE', method_id='netapp.projects.locations.volumes.replications.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetappProjectsLocationsVolumesReplicationsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Describe a replication for a volume.

      Args:
        request: (NetappProjectsLocationsVolumesReplicationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Replication) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}/replications/{replicationsId}', http_method='GET', method_id='netapp.projects.locations.volumes.replications.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetappProjectsLocationsVolumesReplicationsGetRequest', response_type_name='Replication', supports_download=False)

    def List(self, request, global_params=None):
        """Returns descriptions of all replications for a volume.

      Args:
        request: (NetappProjectsLocationsVolumesReplicationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListReplicationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}/replications', http_method='GET', method_id='netapp.projects.locations.volumes.replications.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/replications', request_field='', request_type_name='NetappProjectsLocationsVolumesReplicationsListRequest', response_type_name='ListReplicationsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the settings of a specific replication.

      Args:
        request: (NetappProjectsLocationsVolumesReplicationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}/replications/{replicationsId}', http_method='PATCH', method_id='netapp.projects.locations.volumes.replications.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='replication', request_type_name='NetappProjectsLocationsVolumesReplicationsPatchRequest', response_type_name='Operation', supports_download=False)

    def Resume(self, request, global_params=None):
        """Resume Cross Region Replication.

      Args:
        request: (NetappProjectsLocationsVolumesReplicationsResumeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Resume')
        return self._RunMethod(config, request, global_params=global_params)
    Resume.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}/replications/{replicationsId}:resume', http_method='POST', method_id='netapp.projects.locations.volumes.replications.resume', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:resume', request_field='resumeReplicationRequest', request_type_name='NetappProjectsLocationsVolumesReplicationsResumeRequest', response_type_name='Operation', supports_download=False)

    def ReverseDirection(self, request, global_params=None):
        """Reverses direction of replication. Source becomes destination and destination becomes source.

      Args:
        request: (NetappProjectsLocationsVolumesReplicationsReverseDirectionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ReverseDirection')
        return self._RunMethod(config, request, global_params=global_params)
    ReverseDirection.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}/replications/{replicationsId}:reverseDirection', http_method='POST', method_id='netapp.projects.locations.volumes.replications.reverseDirection', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:reverseDirection', request_field='reverseReplicationDirectionRequest', request_type_name='NetappProjectsLocationsVolumesReplicationsReverseDirectionRequest', response_type_name='Operation', supports_download=False)

    def Stop(self, request, global_params=None):
        """Stop Cross Region Replication.

      Args:
        request: (NetappProjectsLocationsVolumesReplicationsStopRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Stop')
        return self._RunMethod(config, request, global_params=global_params)
    Stop.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}/replications/{replicationsId}:stop', http_method='POST', method_id='netapp.projects.locations.volumes.replications.stop', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:stop', request_field='stopReplicationRequest', request_type_name='NetappProjectsLocationsVolumesReplicationsStopRequest', response_type_name='Operation', supports_download=False)