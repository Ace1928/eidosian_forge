from keystoneauth1.extras import _saml2
from keystoneauth1 import loading
class ADFSPassword(loading.BaseFederationLoader):

    @property
    def plugin_class(self):
        return _saml2.V3ADFSPassword

    @property
    def available(self):
        return _saml2._V3_ADFS_AVAILABLE

    def get_options(self):
        options = super(ADFSPassword, self).get_options()
        options.extend([loading.Opt('identity-provider-url', required=True, help='An Identity Provider URL, where the SAML authentication request will be sent.'), loading.Opt('service-provider-endpoint', required=True, help="Service Provider's Endpoint"), loading.Opt('service-provider-entity-id', required=True, help="Service Provider's SAML Entity ID"), loading.Opt('username', help='Username', required=True), loading.Opt('password', secret=True, required=True, help='Password')])
        return options