from keystoneclient.auth.identity import v3
class OAuthMethod(v3.AuthMethod):
    """OAuth based authentication method.

    :param string consumer_key: Consumer key.
    :param string consumer_secret: Consumer secret.
    :param string access_key: Access token key.
    :param string access_secret: Access token secret.
    """
    _method_parameters = ['consumer_key', 'consumer_secret', 'access_key', 'access_secret']

    def __init__(self, **kwargs):
        super(OAuthMethod, self).__init__(**kwargs)
        if oauth1 is None:
            raise NotImplementedError('optional package oauthlib is not installed')

    def get_auth_data(self, session, auth, headers, **kwargs):
        oauth_client = oauth1.Client(self.consumer_key, client_secret=self.consumer_secret, resource_owner_key=self.access_key, resource_owner_secret=self.access_secret, signature_method=oauth1.SIGNATURE_HMAC)
        o_url, o_headers, o_body = oauth_client.sign(auth.token_url, http_method='POST')
        headers.update(o_headers)
        return ('oauth1', {})