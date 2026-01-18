import os
import sys
import time
import errno
import base64
import logging
import datetime
import urllib.parse
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from libcloud.utils.py3 import b, httplib, urlparse, urlencode
from libcloud.common.base import BaseDriver, JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError
from libcloud.utils.connection import get_response_object
class GoogleInstalledAppAuthConnection(GoogleBaseAuthConnection):
    """Authentication connection for "Installed Application" authentication."""
    _state = 'Libcloud Request'

    def get_code(self):
        """
        Give the user a URL that they can visit to authenticate.

        Mocked in libcloud.test.common.google.GoogleTestCase.

        :return:  Code supplied by the user after authenticating
        :rtype:   ``str``
        """
        auth_params = {'response_type': 'code', 'client_id': self.user_id, 'redirect_uri': self._redirect_uri_with_port, 'scope': self.scopes, 'state': self._state}
        if self.login_hint:
            auth_params['login_hint'] = self.login_hint
        data = urlencode(auth_params)
        url = 'https://{}{}?{}'.format(self.host, self.auth_path, data)
        print('\nPlease Go to the following URL and sign in:')
        print(url)
        code = self._receive_code_through_local_loopback()
        return code

    def get_new_token(self):
        """
        Get a new token. Generally used when no previous token exists or there
        is no refresh token

        :return:  Dictionary containing token information
        :rtype:   ``dict``
        """
        code = self.get_code()
        token_request = {'code': code, 'client_id': self.user_id, 'client_secret': self.key, 'redirect_uri': self._redirect_uri_with_port, 'grant_type': 'authorization_code'}
        return self._token_request(token_request)

    def refresh_token(self, token_info):
        """
        Use the refresh token supplied in the token info to get a new token.

        :param  token_info: Dictionary containing current token information
        :type   token_info: ``dict``

        :return:  A dictionary containing updated token information.
        :rtype:   ``dict``
        """
        if 'refresh_token' not in token_info:
            return self.get_new_token()
        refresh_request = {'refresh_token': token_info['refresh_token'], 'client_id': self.user_id, 'client_secret': self.key, 'grant_type': 'refresh_token'}
        new_token = self._token_request(refresh_request)
        if 'refresh_token' not in new_token:
            new_token['refresh_token'] = token_info['refresh_token']
        return new_token

    @property
    def _redirect_uri_with_port(self):
        return self.redirect_uri + ':' + str(self.redirect_uri_port)

    def _receive_code_through_local_loopback(self):
        """
        Start a local HTTP server that listens to a single GET request that is expected to be made
        by the loopback in the sign-in process and stops again afterwards.
        See https://developers.google.com/identity/protocols/oauth2/native-app#redirect-uri_loopback

        :return: The access code that was extracted from the local loopback GET request
        :rtype: ``str``
        """
        access_code = None

        class AccessCodeReceiver(BaseHTTPRequestHandler):

            def do_GET(self_):
                query = urlparse.urlparse(self_.path).query
                query_components = dict((qc.split('=') for qc in query.split('&')))
                if 'state' in query_components and query_components['state'] != urllib.parse.quote(self._state):
                    raise ValueError("States do not match: {} != {}, can't trust authentication".format(self._state, query_components['state']))
                nonlocal access_code
                access_code = query_components['code']
                self_.send_response(200)
                self_.send_header('Content-type', 'text/html')
                self_.end_headers()
                self_.wfile.write(b'<html><head><title>Libcloud Sign-In</title></head>')
                self_.wfile.write(b'<body><p>You can now close this tab</p>')
        if '127.0.0.1' in self.redirect_uri or '[::1]' in self.redirect_uri or 'localhost' in self.redirect_uri:
            server_address = ('localhost', self.redirect_uri_port)
        else:
            server_address = (self.redirect_uri, self.redirect_uri_port)
        server = HTTPServer(server_address=server_address, RequestHandlerClass=AccessCodeReceiver)
        server.handle_request()
        if access_code is None:
            raise RuntimeError('Could not receive OAuth2 code: could not extract code though loopback')
        return access_code