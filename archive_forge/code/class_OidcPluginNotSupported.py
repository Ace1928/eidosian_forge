from keystoneauth1.exceptions import auth_plugins
class OidcPluginNotSupported(auth_plugins.AuthPluginException):
    message = 'OpenID Connect grant type not supported by provider.'