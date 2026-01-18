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
class RemoteLoginWithAuthProxyFlow(InstalledAppFlow):
    """Flow to authorize gcloud on a machine without access to web browsers.

  Out-of-band flow (OobFlow) is deprecated. gcloud in
  environments without access to browsers (eg. access via ssh) can use this
  flow to authorize gcloud. This flow will print a url which the user has to
  copy to a browser in any machine and perform authorization. After the
  authorization, the user is redirected to gcloud's auth proxy which displays
  the auth code. User copies the auth code back to gcloud to continue the
  process (exchanging auth code for the refresh/access tokens).
  """

    def __init__(self, oauth2session, client_type, client_config, redirect_uri=None, code_verifier=None, autogenerate_code_verifier=False):
        super(RemoteLoginWithAuthProxyFlow, self).__init__(oauth2session, client_type, client_config, redirect_uri=redirect_uri, code_verifier=code_verifier, autogenerate_code_verifier=autogenerate_code_verifier, require_local_server=False)

    def _Run(self, **kwargs):
        """Run the flow using the console strategy.

    The console strategy instructs the user to open the authorization URL
    in their browser. Once the authorization is complete the authorization
    server will give the user a code. The user then must copy & paste this
    code into the application. The code is then exchanged for a token.

    Args:
        **kwargs: Additional keyword arguments passed through to
          "authorization_url".

    Returns:
        google.oauth2.credentials.Credentials: The OAuth 2.0 credentials
          for the user.
    """
        kwargs.setdefault('prompt', 'consent')
        kwargs.setdefault('token_usage', 'remote')
        auth_url, _ = self.authorization_url(**kwargs)
        authorization_prompt_message = 'Go to the following link in your browser:\n\n    {url}\n'
        code = PromptForAuthCode(authorization_prompt_message, auth_url, self.client_config)
        self.fetch_token(code=code, include_client_id=self.include_client_id, verify=None)
        return self.credentials