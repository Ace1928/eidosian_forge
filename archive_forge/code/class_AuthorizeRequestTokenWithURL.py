from __future__ import print_function
import httplib2
import json
import os
from select import select
import stat
from sys import stdin
import time
import webbrowser
from base64 import (
from six.moves.urllib.parse import parse_qs
from lazr.restfulclient.errors import HTTPError
from lazr.restfulclient.authorize.oauth import (
from launchpadlib import uris
class AuthorizeRequestTokenWithURL(RequestTokenAuthorizationEngine):
    """Authorize using a URL.

    This authorizer simply shows the URL for the user to open for
    authorization, and waits until the server responds.
    """
    WAITING_FOR_USER = 'Please open this authorization page:\n (%s)\nin your browser. Use your browser to authorize\nthis program to access Launchpad on your behalf.'
    WAITING_FOR_LAUNCHPAD = 'Press Enter after authorizing in your browser.'

    def output(self, message):
        """Display a message.

        By default, prints the message to standard output. The message
        does not require any user interaction--it's solely
        informative.
        """
        print(message)

    def notify_end_user_authorization_url(self, authorization_url):
        """Notify the end-user of the URL."""
        self.output(self.WAITING_FOR_USER % authorization_url)

    def check_end_user_authorization(self, credentials):
        """Check if the end-user authorized"""
        try:
            credentials.exchange_request_token_for_access_token(self.web_root)
        except HTTPError as e:
            if e.response.status == 403:
                raise EndUserDeclinedAuthorization(e.content)
            else:
                if e.response.status != 401:
                    print('Unexpected response from Launchpad:')
                    print(e)
                raise EndUserNoAuthorization(e.content)
        return credentials.access_token is not None

    def wait_for_end_user_authorization(self, credentials):
        """Wait for the end-user to authorize"""
        self.output(self.WAITING_FOR_LAUNCHPAD)
        stdin.readline()
        self.check_end_user_authorization(credentials)

    def make_end_user_authorize_token(self, credentials, request_token):
        """Have the end-user authorize the token using a URL."""
        authorization_url = self.authorization_url(request_token)
        self.notify_end_user_authorization_url(authorization_url)
        self.wait_for_end_user_authorization(credentials)