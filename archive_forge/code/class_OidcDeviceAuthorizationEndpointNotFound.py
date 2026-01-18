from keystoneauth1.exceptions import auth_plugins
class OidcDeviceAuthorizationEndpointNotFound(auth_plugins.AuthPluginException):
    message = 'OpenID Connect device authorization endpoint not provided.'