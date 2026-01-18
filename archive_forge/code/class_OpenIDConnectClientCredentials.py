from keystoneauth1 import exceptions
from keystoneauth1 import identity
from keystoneauth1 import loading
class OpenIDConnectClientCredentials(_OpenIDConnectBase):

    @property
    def plugin_class(self):
        return identity.V3OidcClientCredentials

    def get_options(self):
        options = super(OpenIDConnectClientCredentials, self).get_options()
        return options