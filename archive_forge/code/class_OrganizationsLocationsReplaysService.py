from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.policysimulator.v1beta import policysimulator_v1beta_messages as messages
class OrganizationsLocationsReplaysService(base_api.BaseApiService):
    """Service class for the organizations_locations_replays resource."""
    _NAME = 'organizations_locations_replays'

    def __init__(self, client):
        super(PolicysimulatorV1beta.OrganizationsLocationsReplaysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates and starts a Replay using the given ReplayConfig.

      Args:
        request: (PolicysimulatorOrganizationsLocationsReplaysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/organizations/{organizationsId}/locations/{locationsId}/replays', http_method='POST', method_id='policysimulator.organizations.locations.replays.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1beta/{+parent}/replays', request_field='googleCloudPolicysimulatorV1betaReplay', request_type_name='PolicysimulatorOrganizationsLocationsReplaysCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified Replay. Each `Replay` is available for at least 7 days.

      Args:
        request: (PolicysimulatorOrganizationsLocationsReplaysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudPolicysimulatorV1betaReplay) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/organizations/{organizationsId}/locations/{locationsId}/replays/{replaysId}', http_method='GET', method_id='policysimulator.organizations.locations.replays.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='PolicysimulatorOrganizationsLocationsReplaysGetRequest', response_type_name='GoogleCloudPolicysimulatorV1betaReplay', supports_download=False)

    def List(self, request, global_params=None):
        """Lists each Replay in a project, folder, or organization. Each `Replay` is available for at least 7 days.

      Args:
        request: (PolicysimulatorOrganizationsLocationsReplaysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudPolicysimulatorV1betaListReplaysResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/organizations/{organizationsId}/locations/{locationsId}/replays', http_method='GET', method_id='policysimulator.organizations.locations.replays.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/replays', request_field='', request_type_name='PolicysimulatorOrganizationsLocationsReplaysListRequest', response_type_name='GoogleCloudPolicysimulatorV1betaListReplaysResponse', supports_download=False)