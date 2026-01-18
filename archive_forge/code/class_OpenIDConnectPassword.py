from keystoneauth1 import exceptions
from keystoneauth1 import identity
from keystoneauth1 import loading
class OpenIDConnectPassword(_OpenIDConnectBase):

    @property
    def plugin_class(self):
        return identity.V3OidcPassword

    def get_options(self):
        options = super(OpenIDConnectPassword, self).get_options()
        options.extend([loading.Opt('username', help='Username', required=True), loading.Opt('password', secret=True, help='Password', required=True)])
        return options