from keystoneauth1 import exceptions
from keystoneauth1 import identity
from keystoneauth1 import loading
class _OpenIDConnectBase(loading.BaseFederationLoader):

    def load_from_options(self, **kwargs):
        if not (kwargs.get('access_token_endpoint') or kwargs.get('discovery_endpoint')):
            m = "You have to specify either an 'access-token-endpoint' or a 'discovery-endpoint'."
            raise exceptions.OptionError(m)
        return super(_OpenIDConnectBase, self).load_from_options(**kwargs)

    def get_options(self):
        options = super(_OpenIDConnectBase, self).get_options()
        options.extend([loading.Opt('client-id', help='OAuth 2.0 Client ID'), loading.Opt('client-secret', secret=True, help='OAuth 2.0 Client Secret'), loading.Opt('openid-scope', default='openid profile', dest='scope', help='OpenID Connect scope that is requested from authorization server. Note that the OpenID Connect specification states that "openid" must be always specified.'), loading.Opt('access-token-endpoint', help='OpenID Connect Provider Token Endpoint. Note that if a discovery document is being passed this option will override the endpoint provided by the server in the discovery document.'), loading.Opt('discovery-endpoint', help='OpenID Connect Discovery Document URL. The discovery document will be used to obtain the values of the access token endpoint and the authentication endpoint. This URL should look like https://idp.example.org/.well-known/openid-configuration'), loading.Opt('access-token-type', help='OAuth 2.0 Authorization Server Introspection token type, it is used to decide which type of token will be used when processing token introspection. Valid values are: "access_token" or "id_token"')])
        return options