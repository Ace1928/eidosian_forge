import logging
from keystoneauth1.identity import v3
class OAuth1Method(v3.AuthMethod):
    """OAuth based authentication method.

    :param string consumer_key: Consumer key.
    :param string consumer_secret: Consumer secret.
    :param string access_key: Access token key.
    :param string access_secret: Access token secret.
    """
    _method_parameters = ['consumer_key', 'consumer_secret', 'access_key', 'access_secret']

    def get_auth_data(self, session, auth, headers, **kwargs):
        oauth_client = oauth1.Client(self.consumer_key, client_secret=self.consumer_secret, resource_owner_key=self.access_key, resource_owner_secret=self.access_secret, signature_method=oauth1.SIGNATURE_HMAC)
        o_url, o_headers, o_body = oauth_client.sign(auth.token_url, http_method='POST')
        headers.update(o_headers)
        return ('oauth1', {})

    def get_cache_id_elements(self):
        return dict((('oauth1_%s' % p, getattr(self, p)) for p in self._method_parameters))