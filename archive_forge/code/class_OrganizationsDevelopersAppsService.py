from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsDevelopersAppsService(base_api.BaseApiService):
    """Service class for the organizations_developers_apps resource."""
    _NAME = 'organizations_developers_apps'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsDevelopersAppsService, self).__init__(client)
        self._upload_configs = {}

    def Attributes(self, request, global_params=None):
        """Updates attributes for a developer app. This API replaces the current attributes with those specified in the request.

      Args:
        request: (ApigeeOrganizationsDevelopersAppsAttributesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Attributes) The response message.
      """
        config = self.GetMethodConfig('Attributes')
        return self._RunMethod(config, request, global_params=global_params)
    Attributes.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/apps/{appsId}/attributes', http_method='POST', method_id='apigee.organizations.developers.apps.attributes', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}/attributes', request_field='googleCloudApigeeV1Attributes', request_type_name='ApigeeOrganizationsDevelopersAppsAttributesRequest', response_type_name='GoogleCloudApigeeV1Attributes', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates an app associated with a developer. This API associates the developer app with the specified API product and auto-generates an API key for the app to use in calls to API proxies inside that API product. The `name` is the unique ID of the app that you can use in API calls. The `DisplayName` (set as an attribute) appears in the UI. If you don't set the `DisplayName` attribute, the `name` appears in the UI.

      Args:
        request: (ApigeeOrganizationsDevelopersAppsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperApp) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/apps', http_method='POST', method_id='apigee.organizations.developers.apps.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/apps', request_field='googleCloudApigeeV1DeveloperApp', request_type_name='ApigeeOrganizationsDevelopersAppsCreateRequest', response_type_name='GoogleCloudApigeeV1DeveloperApp', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a developer app. **Note**: The delete operation is asynchronous. The developer app is deleted immediately, but its associated resources, such as app keys or access tokens, may take anywhere from a few seconds to a few minutes to be deleted.

      Args:
        request: (ApigeeOrganizationsDevelopersAppsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperApp) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/apps/{appsId}', http_method='DELETE', method_id='apigee.organizations.developers.apps.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsDevelopersAppsDeleteRequest', response_type_name='GoogleCloudApigeeV1DeveloperApp', supports_download=False)

    def GenerateKeyPairOrUpdateDeveloperAppStatus(self, request, global_params=None):
        """Manages access to a developer app by enabling you to: * Approve or revoke a developer app * Generate a new consumer key and secret for a developer app To approve or revoke a developer app, set the `action` query parameter to `approve` or `revoke`, respectively, and the `Content-Type` header to `application/octet-stream`. If a developer app is revoked, none of its API keys are valid for API calls even though the keys are still approved. If successful, the API call returns the following HTTP status code: `204 No Content` To generate a new consumer key and secret for a developer app, pass the new key/secret details. Rather than replace an existing key, this API generates a new key. In this case, multiple key pairs may be associated with a single developer app. Each key pair has an independent status (`approve` or `revoke`) and expiration time. Any approved, non-expired key can be used in an API call. For example, if you're using API key rotation, you can generate new keys with expiration times that overlap keys that are going to expire. You might also generate a new consumer key/secret if the security of the original key/secret is compromised. The `keyExpiresIn` property defines the expiration time for the API key in milliseconds. If you don't set this property or set it to `-1`, the API key never expires. **Notes**: * When generating a new key/secret, this API replaces the existing attributes, notes, and callback URLs with those specified in the request. Include or exclude any existing information that you want to retain or delete, respectively. * To migrate existing consumer keys and secrets to hybrid from another system, see the CreateDeveloperAppKey API.

      Args:
        request: (ApigeeOrganizationsDevelopersAppsGenerateKeyPairOrUpdateDeveloperAppStatusRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperApp) The response message.
      """
        config = self.GetMethodConfig('GenerateKeyPairOrUpdateDeveloperAppStatus')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateKeyPairOrUpdateDeveloperAppStatus.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/apps/{appsId}', http_method='POST', method_id='apigee.organizations.developers.apps.generateKeyPairOrUpdateDeveloperAppStatus', ordered_params=['name'], path_params=['name'], query_params=['action'], relative_path='v1/{+name}', request_field='googleCloudApigeeV1DeveloperApp', request_type_name='ApigeeOrganizationsDevelopersAppsGenerateKeyPairOrUpdateDeveloperAppStatusRequest', response_type_name='GoogleCloudApigeeV1DeveloperApp', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the details for a developer app.

      Args:
        request: (ApigeeOrganizationsDevelopersAppsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperApp) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/apps/{appsId}', http_method='GET', method_id='apigee.organizations.developers.apps.get', ordered_params=['name'], path_params=['name'], query_params=['entity', 'query'], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsDevelopersAppsGetRequest', response_type_name='GoogleCloudApigeeV1DeveloperApp', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all apps created by a developer in an Apigee organization. Optionally, you can request an expanded view of the developer apps. A maximum of 100 developer apps are returned per API call. You can paginate the list of deveoper apps returned using the `startKey` and `count` query parameters.

      Args:
        request: (ApigeeOrganizationsDevelopersAppsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListDeveloperAppsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/apps', http_method='GET', method_id='apigee.organizations.developers.apps.list', ordered_params=['parent'], path_params=['parent'], query_params=['count', 'expand', 'shallowExpand', 'startKey'], relative_path='v1/{+parent}/apps', request_field='', request_type_name='ApigeeOrganizationsDevelopersAppsListRequest', response_type_name='GoogleCloudApigeeV1ListDeveloperAppsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the details for a developer app. In addition, you can add an API product to a developer app and automatically generate an API key for the app to use when calling APIs in the API product. If you want to use an existing API key for the API product, add the API product to the API key using the UpdateDeveloperAppKey API. Using this API, you cannot update the following: * App name as it is the primary key used to identify the app and cannot be changed. * Scopes associated with the app. Instead, use the ReplaceDeveloperAppKey API. This API replaces the existing attributes with those specified in the request. Include or exclude any existing attributes that you want to retain or delete, respectively.

      Args:
        request: (GoogleCloudApigeeV1DeveloperApp) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperApp) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/apps/{appsId}', http_method='PUT', method_id='apigee.organizations.developers.apps.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='GoogleCloudApigeeV1DeveloperApp', response_type_name='GoogleCloudApigeeV1DeveloperApp', supports_download=False)