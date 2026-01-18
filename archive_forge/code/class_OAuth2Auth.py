from __future__ import absolute_import
import boto.auth_handler
from gcs_oauth2_boto_plugin import oauth2_client
from gcs_oauth2_boto_plugin import oauth2_helper
class OAuth2Auth(boto.auth_handler.AuthHandler):
    """AuthHandler for working with OAuth2 user account credentials."""
    capability = ['google-oauth2', 's3']

    def __init__(self, path, config, provider):
        self.oauth2_client = None
        if provider.name == 'google':
            if config.has_option('Credentials', 'gs_oauth2_refresh_token'):
                self.oauth2_client = oauth2_helper.OAuth2ClientFromBotoConfig(config)
            elif config.has_option('GoogleCompute', 'service_account'):
                self.oauth2_client = oauth2_client.CreateOAuth2GCEClient()
        if not self.oauth2_client:
            raise boto.auth_handler.NotReadyToAuthenticate()

    def add_auth(self, http_request):
        http_request.headers['Authorization'] = self.oauth2_client.GetAuthorizationHeader()