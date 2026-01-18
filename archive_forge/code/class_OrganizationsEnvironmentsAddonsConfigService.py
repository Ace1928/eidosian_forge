from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsAddonsConfigService(base_api.BaseApiService):
    """Service class for the organizations_environments_addonsConfig resource."""
    _NAME = 'organizations_environments_addonsConfig'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsAddonsConfigService, self).__init__(client)
        self._upload_configs = {}

    def SetAddonEnablement(self, request, global_params=None):
        """Updates an add-on enablement status of an environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsAddonsConfigSetAddonEnablementRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('SetAddonEnablement')
        return self._RunMethod(config, request, global_params=global_params)
    SetAddonEnablement.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/addonsConfig:setAddonEnablement', http_method='POST', method_id='apigee.organizations.environments.addonsConfig.setAddonEnablement', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:setAddonEnablement', request_field='googleCloudApigeeV1SetAddonEnablementRequest', request_type_name='ApigeeOrganizationsEnvironmentsAddonsConfigSetAddonEnablementRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)