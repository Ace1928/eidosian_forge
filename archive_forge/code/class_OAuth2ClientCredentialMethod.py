import requests.auth
from keystoneauth1.exceptions import ClientException
from keystoneauth1.identity.v3 import base
class OAuth2ClientCredentialMethod(base.AuthMethod):
    """An auth method to fetch a token via an OAuth2.0 client credential.

    :param string oauth2_endpoint: OAuth2.0 endpoint.
    :param string oauth2_client_id: OAuth2.0 client credential id.
    :param string oauth2_client_secret: OAuth2.0 client credential secret.
    """
    _method_parameters = ['oauth2_endpoint', 'oauth2_client_id', 'oauth2_client_secret']

    def get_auth_data(self, session, auth, headers, **kwargs):
        """Return the authentication section of an auth plugin.

        :param session: The communication session.
        :type session: keystoneauth1.session.Session
        :param base.Auth auth: The auth plugin calling the method.
        :param dict headers: The headers that will be sent with the auth
                             request if a plugin needs to add to them.
        :return: The identifier of this plugin and a dict of authentication
                 data for the auth type.
        :rtype: tuple(string, dict)
        """
        auth_data = {'id': self.oauth2_client_id, 'secret': self.oauth2_client_secret}
        return ('application_credential', auth_data)

    def get_cache_id_elements(self):
        """Get the elements for this auth method that make it unique.

        These elements will be used as part of the
        :py:meth:`keystoneauth1.plugin.BaseIdentityPlugin.get_cache_id` to
        allow caching of the auth plugin.

        Plugins should override this if they want to allow caching of their
        state.

        To avoid collision or overrides the keys of the returned dictionary
        should be prefixed with the plugin identifier. For example the password
        plugin returns its username value as 'password_username'.
        """
        return dict((('oauth2_client_credential_%s' % p, getattr(self, p)) for p in self._method_parameters))