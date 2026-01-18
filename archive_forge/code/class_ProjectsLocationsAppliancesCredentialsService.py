from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.transferappliance.v1alpha1 import transferappliance_v1alpha1_messages as messages
class ProjectsLocationsAppliancesCredentialsService(base_api.BaseApiService):
    """Service class for the projects_locations_appliances_credentials resource."""
    _NAME = 'projects_locations_appliances_credentials'

    def __init__(self, client):
        super(TransferapplianceV1alpha1.ProjectsLocationsAppliancesCredentialsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets Credentials of the appliance.

      Args:
        request: (TransferapplianceProjectsLocationsAppliancesCredentialsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Credential) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/appliances/{appliancesId}/credentials/{credentialsId}', http_method='GET', method_id='transferappliance.projects.locations.appliances.credentials.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='TransferapplianceProjectsLocationsAppliancesCredentialsGetRequest', response_type_name='Credential', supports_download=False)