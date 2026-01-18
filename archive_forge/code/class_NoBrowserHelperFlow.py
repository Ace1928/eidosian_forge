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
class NoBrowserHelperFlow(InstalledAppFlow):
    """Helper flow for the NoBrowserFlow to help another gcloud to authorize.

  This flow takes the authorization parameters (i.e. requested scopes) generated
  by the NoBrowserFlow and launches the browser for users to authorize.
  After users authorize, print the authorization response which will be taken
  by NoBrowserFlow to continue the login process
  (exchanging for refresh/access token).
  """
    _COPY_AUTH_RESPONSE_INSTRUCTION = 'Copy the following line back to the gcloud CLI waiting to continue the login flow.'
    _COPY_AUTH_RESPONSE_WARNING = '{bold}WARNING: The following line enables access to your Google Cloud resources. Only copy it to the trusted machine that you ran the `{command} --no-browser` command on earlier.{normal}'
    _PROMPT_TO_CONTINUE_MSG = 'DO NOT PROCEED UNLESS YOU ARE BOOTSTRAPPING GCLOUD ON A TRUSTED MACHINE WITHOUT A WEB BROWSER AND THE ABOVE COMMAND WAS THE OUTPUT OF `{command} --no-browser` FROM THE TRUSTED MACHINE.'

    def __init__(self, oauth2session, client_type, client_config, redirect_uri=None, code_verifier=None, autogenerate_code_verifier=False):
        super(NoBrowserHelperFlow, self).__init__(oauth2session, client_type, client_config, redirect_uri=redirect_uri, code_verifier=code_verifier, autogenerate_code_verifier=autogenerate_code_verifier, require_local_server=True)
        self.partial_auth_url = None

    @property
    def _for_adc(self):
        client_id = UrlManager(self.partial_auth_url).GetQueryParam('client_id')
        return client_id != config.CLOUDSDK_CLIENT_ID

    def _PrintCopyInstruction(self, auth_response):
        con = console_attr.GetConsoleAttr()
        log.status.write(self._COPY_AUTH_RESPONSE_INSTRUCTION + ' ')
        log.status.Print(self._COPY_AUTH_RESPONSE_WARNING.format(bold=con.GetFontCode(bold=True), command=self._target_command, normal=con.GetFontCode()))
        log.status.write('\n')
        log.status.Print(auth_response)

    def _ShouldContinue(self):
        """Ask users to confirm before actually running the flow."""
        return console_io.PromptContinue(self._PROMPT_TO_CONTINUE_MSG.format(command=self._target_command), prompt_string='Proceed', default=False)

    def _Run(self, **kwargs):
        self.partial_auth_url = kwargs.pop('partial_auth_url')
        auth_url_manager = UrlManager(self.partial_auth_url)
        auth_url_manager.UpdateQueryParams([('redirect_uri', self.redirect_uri)] + list(kwargs.items()))
        auth_url = auth_url_manager.GetUrl()
        if not self._ShouldContinue():
            return
        webbrowser.open(auth_url, new=1, autoraise=True)
        authorization_prompt_message = 'Your browser has been opened to visit:\n\n    {url}\n'
        log.err.Print(authorization_prompt_message.format(url=auth_url))
        self.server.handle_request()
        self.server.server_close()
        if not self.app.last_request_uri:
            raise LocalServerTimeoutError('Local server timed out before receiving the redirection request.')
        authorization_response = self.app.last_request_uri.replace('http:', 'https:')
        self._PrintCopyInstruction(authorization_response)