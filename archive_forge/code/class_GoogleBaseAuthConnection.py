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
class GoogleBaseAuthConnection(ConnectionUserAndKey):
    """
    Base class for Google Authentication.  Should be subclassed for specific
    types of authentication.
    """
    driver = GoogleBaseDriver
    responseCls = GoogleResponse
    name = 'Google Auth'
    host = 'accounts.google.com'
    auth_path = '/o/oauth2/auth'
    redirect_uri_port = 8087

    def __init__(self, user_id, key=None, scopes=None, redirect_uri='http://127.0.0.1', login_hint=None, **kwargs):
        """
        :param  user_id: The email address (for service accounts) or Client ID
                         (for installed apps) to be used for authentication.
        :type   user_id: ``str``

        :param  key: The RSA Key (for service accounts) or file path containing
                     key or Client Secret (for installed apps) to be used for
                     authentication.
        :type   key: ``str``

        :param  scopes: A list of urls defining the scope of authentication
                       to grant.
        :type   scopes: ``list``

        :keyword  redirect_uri: The Redirect URI for the authentication
                                request.  See Google OAUTH2 documentation for
                                more info.
        :type     redirect_uri: ``str``

        :keyword  login_hint: Login hint for authentication request.  Useful
                              for Installed Application authentication.
        :type     login_hint: ``str``
        """
        scopes = scopes or []
        self.scopes = ' '.join(scopes)
        self.redirect_uri = redirect_uri
        self.login_hint = login_hint
        super().__init__(user_id, key, **kwargs)

    def add_default_headers(self, headers):
        """
        Add defaults for 'Content-Type' and 'Host' headers.
        """
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
        headers['Host'] = self.host
        return headers

    def _token_request(self, request_body):
        """
        Return an updated token from a token request body.

        :param  request_body: A dictionary of values to send in the body of the
                              token request.
        :type   request_body: ``dict``

        :return:  A dictionary with updated token information
        :rtype:   ``dict``
        """
        data = urlencode(request_body)
        try:
            response = self.request('/o/oauth2/token', method='POST', data=data)
        except AttributeError:
            raise GoogleAuthError('Invalid authorization response, please check your credentials and time drift.')
        token_info = response.object
        if 'expires_in' in token_info:
            expire_time = _utcnow() + datetime.timedelta(seconds=token_info['expires_in'])
            token_info['expire_time'] = _utc_timestamp(expire_time)
        return token_info

    def refresh_token(self, token_info):
        """
        Refresh the current token.

        Fetch an updated refresh token from internal metadata service.

        :param  token_info: Dictionary containing token information.
                            (Not used, but here for compatibility)
        :type   token_info: ``dict``

        :return:  A dictionary containing updated token information.
        :rtype:   ``dict``
        """
        return self.get_new_token()