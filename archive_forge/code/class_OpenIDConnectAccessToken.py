from keystoneauth1 import exceptions
from keystoneauth1 import identity
from keystoneauth1 import loading
class OpenIDConnectAccessToken(loading.BaseFederationLoader):

    @property
    def plugin_class(self):
        return identity.V3OidcAccessToken

    def get_options(self):
        options = super(OpenIDConnectAccessToken, self).get_options()
        options.extend([loading.Opt('access-token', secret=True, required=True, help='OAuth 2.0 Access Token')])
        return options