from keystoneauth1.exceptions import auth_plugins
class OidcDeviceAuthorizationTimeOut(auth_plugins.AuthPluginException):
    message = 'Timeout for OpenID Connect device authorization.'