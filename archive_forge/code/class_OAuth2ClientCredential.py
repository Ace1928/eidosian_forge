import requests.auth
from keystoneauth1.exceptions import ClientException
from keystoneauth1.identity.v3 import base
class OAuth2ClientCredential(base.AuthConstructor):
    """A plugin for authenticating via an OAuth2.0 client credential.

    :param string auth_url: Identity service endpoint for authentication.
    :param string oauth2_endpoint: OAuth2.0 endpoint.
    :param string oauth2_client_id: OAuth2.0 client credential id.
    :param string oauth2_client_secret: OAuth2.0 client credential secret.
    """
    _auth_method_class = OAuth2ClientCredentialMethod

    def __init__(self, auth_url, *args, **kwargs):
        super(OAuth2ClientCredential, self).__init__(auth_url, *args, **kwargs)
        self._oauth2_endpoint = kwargs['oauth2_endpoint']
        self._oauth2_client_id = kwargs['oauth2_client_id']
        self._oauth2_client_secret = kwargs['oauth2_client_secret']

    def get_headers(self, session, **kwargs):
        """Fetch authentication headers for message.

        :param session: The session object that the auth_plugin belongs to.
        :type session: keystoneauth1.session.Session

        :returns: Headers that are set to authenticate a message or None for
                  failure. Note that when checking this value that the empty
                  dict is a valid, non-failure response.
        :rtype: dict
        """
        headers = super(OAuth2ClientCredential, self).get_headers(session, **kwargs)
        data = {'grant_type': 'client_credentials'}
        auth = requests.auth.HTTPBasicAuth(self._oauth2_client_id, self._oauth2_client_secret)
        resp = session.request(self._oauth2_endpoint, 'POST', authenticated=False, raise_exc=False, data=data, requests_auth=auth)
        if resp.status_code == 200:
            oauth2 = resp.json()
            oauth2_token = oauth2['access_token']
            if headers:
                headers['Authorization'] = f'Bearer {oauth2_token}'
            else:
                headers = {'Authorization': f'Bearer {oauth2_token}'}
        else:
            error = resp.json()
            msg = error.get('error_description')
            raise ClientException(msg)
        return headers