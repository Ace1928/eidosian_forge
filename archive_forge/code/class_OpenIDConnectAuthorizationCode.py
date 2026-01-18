from keystoneauth1 import exceptions
from keystoneauth1 import identity
from keystoneauth1 import loading
class OpenIDConnectAuthorizationCode(_OpenIDConnectBase):

    @property
    def plugin_class(self):
        return identity.V3OidcAuthorizationCode

    def get_options(self):
        options = super(OpenIDConnectAuthorizationCode, self).get_options()
        options.extend([loading.Opt('redirect-uri', help='OpenID Connect Redirect URL'), loading.Opt('code', secret=True, required=True, deprecated=[loading.Opt('authorization-code')], help='OAuth 2.0 Authorization Code')])
        return options