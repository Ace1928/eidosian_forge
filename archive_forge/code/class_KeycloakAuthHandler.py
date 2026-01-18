import logging
import os
import pprint
import urllib
import requests
from mistralclient import auth
class KeycloakAuthHandler(auth.AuthHandler):

    def authenticate(self, req, session=None):
        """Performs authentication using Keycloak OpenID Protocol.

        :param req: Request dict containing list of parameters required
            for Keycloak authentication.

            * auth_url: Base authentication url of KeyCloak server (e.g.
                "https://my.keycloak:8443/auth"
            * client_id: Client ID (according to OpenID Connect protocol).
            * client_secret: Client secret (according to OpenID Connect
                protocol).
            * project_name: KeyCloak realm name.
            * username: User name (Optional, if None then access_token must be
                provided).
            * api_key: Password (Optional).
            * access_token: Access token. If passed, username and password are
                not used and this method just validates the token and refreshes
                it if needed (Optional, if None then username must be
                provided).
            * cacert: SSL certificate file (Optional).
            * insecure: If True, SSL certificate is not verified (Optional).

        :param session: Keystone session object. Not used by this plugin.

        """
        if not isinstance(req, dict):
            raise TypeError('The input "req" is not typeof dict.')
        auth_url = req.get('auth_url')
        client_id = req.get('client_id')
        client_secret = req.get('client_secret')
        realm_name = req.get('project_name')
        username = req.get('username')
        password = req.get('api_key')
        access_token = req.get('access_token')
        cacert = req.get('cacert')
        insecure = req.get('insecure', False)
        if not auth_url:
            raise ValueError('Base authentication url is not provided.')
        if not client_id:
            raise ValueError('Client ID is not provided.')
        if not realm_name:
            raise ValueError('Project(realm) name is not provided.')
        if username and access_token:
            raise ValueError("User name and access token can't be provided at the same time.")
        if not username and (not access_token):
            raise ValueError('Either user name or access token must be provided.')
        if access_token:
            response = self._authenticate_with_token(auth_url, client_id, client_secret, access_token, cacert, insecure)
        else:
            response = self._authenticate_with_password(auth_url, client_id, client_secret, realm_name, username, password, cacert, insecure)
        return {'auth_token': response, 'project_id': realm_name}

    @staticmethod
    def _authenticate_with_token(auth_url, client_id, client_secret, auth_token, cacert=None, insecure=None):
        raise NotImplementedError

    @staticmethod
    def _authenticate_with_password(auth_url, client_id, client_secret, realm_name, username, password, cacert=None, insecure=None):
        access_token_endpoint = '%s/realms/%s/protocol/openid-connect/token' % (auth_url, realm_name)
        verify = None
        if urllib.parse.urlparse(access_token_endpoint).scheme == 'https':
            verify = False if insecure else cacert if cacert else True
        body = {'grant_type': 'password', 'username': username, 'password': password, 'client_id': client_id, 'scope': 'profile'}
        if client_secret:
            body['client_secret'] = (client_secret,)
        resp = requests.post(access_token_endpoint, data=body, verify=verify)
        try:
            resp.raise_for_status()
        except Exception as e:
            raise Exception('Failed to get access token:\n %s' % str(e))
        LOG.debug('HTTP response from OIDC provider: %s', pprint.pformat(resp.json()))
        return resp.json()['access_token']