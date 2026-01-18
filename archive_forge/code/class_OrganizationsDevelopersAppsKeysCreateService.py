from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsDevelopersAppsKeysCreateService(base_api.BaseApiService):
    """Service class for the organizations_developers_apps_keys_create resource."""
    _NAME = 'organizations_developers_apps_keys_create'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsDevelopersAppsKeysCreateService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a custom consumer key and secret for a developer app. This is particularly useful if you want to migrate existing consumer keys and secrets to Apigee from another system. Consumer keys and secrets can contain letters, numbers, underscores, and hyphens. No other special characters are allowed. To avoid service disruptions, a consumer key and secret should not exceed 2 KBs each. **Note**: When creating the consumer key and secret, an association to API products will not be made. Therefore, you should not specify the associated API products in your request. Instead, use the UpdateDeveloperAppKey API to make the association after the consumer key and secret are created. If a consumer key and secret already exist, you can keep them or delete them using the DeleteDeveloperAppKey API. **Note**: All keys start out with status=approved, even if status=revoked is passed when the key is created. To revoke a key, use the UpdateDeveloperAppKey API.

      Args:
        request: (ApigeeOrganizationsDevelopersAppsKeysCreateCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperAppKey) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/apps/{appsId}/keys/create', http_method='POST', method_id='apigee.organizations.developers.apps.keys.create.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/keys/create', request_field='googleCloudApigeeV1DeveloperAppKey', request_type_name='ApigeeOrganizationsDevelopersAppsKeysCreateCreateRequest', response_type_name='GoogleCloudApigeeV1DeveloperAppKey', supports_download=False)