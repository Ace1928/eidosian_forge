from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securityposture.v1alpha import securityposture_v1alpha_messages as messages
class OrganizationsLocationsPostureTemplatesService(base_api.BaseApiService):
    """Service class for the organizations_locations_postureTemplates resource."""
    _NAME = 'organizations_locations_postureTemplates'

    def __init__(self, client):
        super(SecuritypostureV1alpha.OrganizationsLocationsPostureTemplatesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a PostureTemplate. User must provide revision_id to retrieve a specific revision of the resource. NOT_FOUND error is returned if the revision_id or the PostureTemplate name does not exist. In case revision_id is not provided then the PostureTemplate with latest revision_id is returned.

      Args:
        request: (SecuritypostureOrganizationsLocationsPostureTemplatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PostureTemplate) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/{locationsId}/postureTemplates/{postureTemplatesId}', http_method='GET', method_id='securityposture.organizations.locations.postureTemplates.get', ordered_params=['name'], path_params=['name'], query_params=['revisionId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='SecuritypostureOrganizationsLocationsPostureTemplatesGetRequest', response_type_name='PostureTemplate', supports_download=False)

    def List(self, request, global_params=None):
        """========================== PostureTemplates ========================== Lists all the PostureTemplates available to the user.

      Args:
        request: (SecuritypostureOrganizationsLocationsPostureTemplatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPostureTemplatesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/{locationsId}/postureTemplates', http_method='GET', method_id='securityposture.organizations.locations.postureTemplates.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/postureTemplates', request_field='', request_type_name='SecuritypostureOrganizationsLocationsPostureTemplatesListRequest', response_type_name='ListPostureTemplatesResponse', supports_download=False)