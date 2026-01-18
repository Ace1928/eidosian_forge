from keystoneauth1.exceptions import auth_plugins
class OidcAuthorizationEndpointNotFound(auth_plugins.AuthPluginException):
    message = 'OpenID Connect authorization endpoint not provided.'