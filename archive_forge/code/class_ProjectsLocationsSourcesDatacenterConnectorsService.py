from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmmigration.v1 import vmmigration_v1_messages as messages
class ProjectsLocationsSourcesDatacenterConnectorsService(base_api.BaseApiService):
    """Service class for the projects_locations_sources_datacenterConnectors resource."""
    _NAME = 'projects_locations_sources_datacenterConnectors'

    def __init__(self, client):
        super(VmmigrationV1.ProjectsLocationsSourcesDatacenterConnectorsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new DatacenterConnector in a given Source.

      Args:
        request: (VmmigrationProjectsLocationsSourcesDatacenterConnectorsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/datacenterConnectors', http_method='POST', method_id='vmmigration.projects.locations.sources.datacenterConnectors.create', ordered_params=['parent'], path_params=['parent'], query_params=['datacenterConnectorId', 'requestId'], relative_path='v1/{+parent}/datacenterConnectors', request_field='datacenterConnector', request_type_name='VmmigrationProjectsLocationsSourcesDatacenterConnectorsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single DatacenterConnector.

      Args:
        request: (VmmigrationProjectsLocationsSourcesDatacenterConnectorsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/datacenterConnectors/{datacenterConnectorsId}', http_method='DELETE', method_id='vmmigration.projects.locations.sources.datacenterConnectors.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesDatacenterConnectorsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single DatacenterConnector.

      Args:
        request: (VmmigrationProjectsLocationsSourcesDatacenterConnectorsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DatacenterConnector) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/datacenterConnectors/{datacenterConnectorsId}', http_method='GET', method_id='vmmigration.projects.locations.sources.datacenterConnectors.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesDatacenterConnectorsGetRequest', response_type_name='DatacenterConnector', supports_download=False)

    def List(self, request, global_params=None):
        """Lists DatacenterConnectors in a given Source.

      Args:
        request: (VmmigrationProjectsLocationsSourcesDatacenterConnectorsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDatacenterConnectorsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/datacenterConnectors', http_method='GET', method_id='vmmigration.projects.locations.sources.datacenterConnectors.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/datacenterConnectors', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesDatacenterConnectorsListRequest', response_type_name='ListDatacenterConnectorsResponse', supports_download=False)

    def UpgradeAppliance(self, request, global_params=None):
        """Upgrades the appliance relate to this DatacenterConnector to the in-place updateable version.

      Args:
        request: (VmmigrationProjectsLocationsSourcesDatacenterConnectorsUpgradeApplianceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('UpgradeAppliance')
        return self._RunMethod(config, request, global_params=global_params)
    UpgradeAppliance.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/datacenterConnectors/{datacenterConnectorsId}:upgradeAppliance', http_method='POST', method_id='vmmigration.projects.locations.sources.datacenterConnectors.upgradeAppliance', ordered_params=['datacenterConnector'], path_params=['datacenterConnector'], query_params=[], relative_path='v1/{+datacenterConnector}:upgradeAppliance', request_field='upgradeApplianceRequest', request_type_name='VmmigrationProjectsLocationsSourcesDatacenterConnectorsUpgradeApplianceRequest', response_type_name='Operation', supports_download=False)