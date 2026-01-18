from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datafusion.v1beta1 import datafusion_v1beta1_messages as messages
class ProjectsLocationsInstancesDnsPeeringsService(base_api.BaseApiService):
    """Service class for the projects_locations_instances_dnsPeerings resource."""
    _NAME = 'projects_locations_instances_dnsPeerings'

    def __init__(self, client):
        super(DatafusionV1beta1.ProjectsLocationsInstancesDnsPeeringsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates DNS peering on the given resource.

      Args:
        request: (DatafusionProjectsLocationsInstancesDnsPeeringsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DnsPeering) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/dnsPeerings', http_method='POST', method_id='datafusion.projects.locations.instances.dnsPeerings.create', ordered_params=['parent'], path_params=['parent'], query_params=['dnsPeeringId'], relative_path='v1beta1/{+parent}/dnsPeerings', request_field='dnsPeering', request_type_name='DatafusionProjectsLocationsInstancesDnsPeeringsCreateRequest', response_type_name='DnsPeering', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes DNS peering on the given resource.

      Args:
        request: (DatafusionProjectsLocationsInstancesDnsPeeringsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/dnsPeerings/{dnsPeeringsId}', http_method='DELETE', method_id='datafusion.projects.locations.instances.dnsPeerings.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='DatafusionProjectsLocationsInstancesDnsPeeringsDeleteRequest', response_type_name='Empty', supports_download=False)

    def List(self, request, global_params=None):
        """Lists DNS peerings for a given resource.

      Args:
        request: (DatafusionProjectsLocationsInstancesDnsPeeringsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDnsPeeringsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/dnsPeerings', http_method='GET', method_id='datafusion.projects.locations.instances.dnsPeerings.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/dnsPeerings', request_field='', request_type_name='DatafusionProjectsLocationsInstancesDnsPeeringsListRequest', response_type_name='ListDnsPeeringsResponse', supports_download=False)