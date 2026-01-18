from keystoneauth1.exceptions import auth_plugins
class OidcGrantTypeMissmatch(auth_plugins.AuthPluginException):
    message = 'Missmatch between OpenID Connect plugin and grant_type argument'