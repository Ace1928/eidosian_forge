from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import contextlib
import select
import socket
import sys
import webbrowser
import wsgiref
from google_auth_oauthlib import flow as google_auth_flow
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as c_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import pkg_resources
from oauthlib.oauth2.rfc6749 import errors as rfc6749_errors
from requests import exceptions as requests_exceptions
import six
from six.moves import input  # pylint: disable=redefined-builtin
from six.moves.urllib import parse
class FullWebFlow(InstalledAppFlow):
    """The complete OAuth 2.0 authorization flow.

  This class supports user account login using "gcloud auth login" with browser.
  Specifically, it does the following:
    1. Try to find an available port for the local server which handles the
       redirect.
    2. Create a WSGI app on the local server which can direct browser to
       Google's confirmation pages for authentication.
  """

    def __init__(self, oauth2session, client_type, client_config, redirect_uri=None, code_verifier=None, autogenerate_code_verifier=False):
        super(FullWebFlow, self).__init__(oauth2session, client_type, client_config, redirect_uri=redirect_uri, code_verifier=code_verifier, autogenerate_code_verifier=autogenerate_code_verifier, require_local_server=True)

    def _Run(self, **kwargs):
        """Run the flow using the server strategy.

    The server strategy instructs the user to open the authorization URL in
    their browser and will attempt to automatically open the URL for them.
    It will start a local web server to listen for the authorization
    response. Once authorization is complete the authorization server will
    redirect the user's browser to the local web server. The web server
    will get the authorization code from the response and shutdown. The
    code is then exchanged for a token.

    Args:
        **kwargs: Additional keyword arguments passed through to
          "authorization_url".

    Returns:
        google.oauth2.credentials.Credentials: The OAuth 2.0 credentials
          for the user.

    Raises:
      LocalServerTimeoutError: If the local server handling redirection timeout
        before receiving the request.
    """
        auth_url, _ = self.authorization_url(**kwargs)
        webbrowser.open(auth_url, new=1, autoraise=True)
        authorization_prompt_message = 'Your browser has been opened to visit:\n\n    {url}\n'
        log.err.Print(authorization_prompt_message.format(url=auth_url))
        self.server.handle_request()
        self.server.server_close()
        if not self.app.last_request_uri:
            raise LocalServerTimeoutError('Local server timed out before receiving the redirection request.')
        authorization_response = self.app.last_request_uri.replace('http:', 'https:')
        self.fetch_token(authorization_response=authorization_response, include_client_id=self.include_client_id, verify=None)
        return self.credentials