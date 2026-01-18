from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class OauthService(base_api.BaseApiService):
    """Service class for the oauth resource."""
    _NAME = 'oauth'

    def __init__(self, client):
        super(CloudbuildV1.OauthService, self).__init__(client)
        self._upload_configs = {}

    def GetRegistration(self, request, global_params=None):
        """Get a URL that a customer should use to initiate an OAuth flow on an external source provider. This API is experimental.

      Args:
        request: (CloudbuildOauthGetRegistrationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OAuthRegistrationURI) The response message.
      """
        config = self.GetMethodConfig('GetRegistration')
        return self._RunMethod(config, request, global_params=global_params)
    GetRegistration.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudbuild.oauth.getRegistration', ordered_params=[], path_params=[], query_params=['authUser', 'csesidx', 'githubEnterpriseConfig', 'hostUrl', 'namespace'], relative_path='v1/oauth/registration', request_field='', request_type_name='CloudbuildOauthGetRegistrationRequest', response_type_name='OAuthRegistrationURI', supports_download=False)

    def ProcessOAuthCallback(self, request, global_params=None):
        """ProcessOAuthCallback fulfills the last leg of the OAuth dance with a source provider. For GitHub this is as defined by https://developer.github.com/apps/building-oauth-apps/authorizing-oauth-apps/#2-users-are-redirected-back-to-your-site-by-github Users will not be able to call this in any meaningful way since they don't have access to the OAuth code used in the exchange. For now, this rpc only supports GitHubEnterprise, but will eventually replace GenerateGitHubAccessToken.

      Args:
        request: (CloudbuildOauthProcessOAuthCallbackRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('ProcessOAuthCallback')
        return self._RunMethod(config, request, global_params=global_params)
    ProcessOAuthCallback.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudbuild.oauth.processOAuthCallback', ordered_params=[], path_params=[], query_params=['code', 'githubEnterpriseConfig', 'hostUrl', 'namespace', 'state'], relative_path='v1/oauth:processOAuthCallback', request_field='', request_type_name='CloudbuildOauthProcessOAuthCallbackRequest', response_type_name='Empty', supports_download=False)