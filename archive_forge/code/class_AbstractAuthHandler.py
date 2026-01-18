import base64
import cgi
import errno
import http.client
import os
import re
import socket
import ssl
import sys
import time
import urllib
import urllib.request
import weakref
from urllib.parse import urlencode, urljoin, urlparse
from ... import __version__ as breezy_version
from ... import config, debug, errors, osutils, trace, transport, ui, urlutils
from ...bzr.smart import medium
from ...trace import mutter, mutter_callsite
from ...transport import ConnectedTransport, NoSuchFile, UnusableRedirect
from . import default_user_agent, ssl
from .response import handle_response
class AbstractAuthHandler(urllib.request.BaseHandler):
    """A custom abstract authentication handler for all http authentications.

    Provides the meat to handle authentication errors and
    preventively set authentication headers after the first
    successful authentication.

    This can be used for http and proxy, as well as for basic, negotiate and
    digest authentications.

    This provides an unified interface for all authentication handlers
    (urllib.request provides far too many with different policies).

    The interaction between this handler and the urllib.request
    framework is not obvious, it works as follow:

    opener.open(request) is called:

    - that may trigger http_request which will add an authentication header
      (self.build_header) if enough info is available.

    - the request is sent to the server,

    - if an authentication error is received self.auth_required is called,
      we acquire the authentication info in the error headers and call
      self.auth_match to check that we are able to try the
      authentication and complete the authentication parameters,

    - we call parent.open(request), that may trigger http_request
      and will add a header (self.build_header), but here we have
      all the required info (keep in mind that the request and
      authentication used in the recursive calls are really (and must be)
      the *same* objects).

    - if the call returns a response, the authentication have been
      successful and the request authentication parameters have been updated.
    """
    scheme: str
    'The scheme as it appears in the server header (lower cased)'
    _max_retry = 3
    "We don't want to retry authenticating endlessly"
    requires_username = True
    'Whether the auth mechanism requires a username.'

    def __init__(self):
        self._retry_count = None

    def _parse_auth_header(self, server_header):
        """Parse the authentication header.

        :param server_header: The value of the header sent by the server
            describing the authenticaion request.

        :return: A tuple (scheme, remainder) scheme being the first word in the
            given header (lower cased), remainder may be None.
        """
        try:
            scheme, remainder = server_header.split(None, 1)
        except ValueError:
            scheme = server_header
            remainder = None
        return (scheme.lower(), remainder)

    def update_auth(self, auth, key, value):
        """Update a value in auth marking the auth as modified if needed"""
        old_value = auth.get(key, None)
        if old_value != value:
            auth[key] = value
            auth['modified'] = True

    def auth_required(self, request, headers):
        """Retry the request if the auth scheme is ours.

        :param request: The request needing authentication.
        :param headers: The headers for the authentication error response.
        :return: None or the response for the authenticated request.
        """
        if self._retry_count is None:
            self._retry_count = 1
        else:
            self._retry_count += 1
            if self._retry_count > self._max_retry:
                self._retry_count = None
                return None
        server_headers = headers.get_all(self.auth_required_header)
        if not server_headers:
            trace.mutter('%s not found', self.auth_required_header)
            return None
        auth = self.get_auth(request)
        auth['modified'] = False
        if auth.get('path', None) is None:
            parsed_url = urlutils.URL.from_string(request.get_full_url())
            self.update_auth(auth, 'protocol', parsed_url.scheme)
            self.update_auth(auth, 'host', parsed_url.host)
            self.update_auth(auth, 'port', parsed_url.port)
            self.update_auth(auth, 'path', parsed_url.path)
        for server_header in server_headers:
            matching_handler = self.auth_match(server_header, auth)
            if matching_handler:
                if request.get_header(self.auth_header, None) is not None and (not auth['modified']):
                    return None
                best_scheme = auth.get('best_scheme', None)
                if best_scheme is None:
                    best_scheme = auth['best_scheme'] = self.scheme
                if best_scheme != self.scheme:
                    continue
                if self.requires_username and auth.get('user', None) is None:
                    return None
                request.connection.cleanup_pipe()
                response = self.parent.open(request)
                if response:
                    self.auth_successful(request, response)
                return response
        return None

    def add_auth_header(self, request, header):
        """Add the authentication header to the request"""
        request.add_unredirected_header(self.auth_header, header)

    def auth_match(self, header, auth):
        """Check that we are able to handle that authentication scheme.

        The request authentication parameters may need to be
        updated with info from the server. Some of these
        parameters, when combined, are considered to be the
        authentication key, if one of them change the
        authentication result may change. 'user' and 'password'
        are exampls, but some auth schemes may have others
        (digest's nonce is an example, digest's nonce_count is a
        *counter-example*). Such parameters must be updated by
        using the update_auth() method.

        :param header: The authentication header sent by the server.
        :param auth: The auth parameters already known. They may be
             updated.
        :returns: True if we can try to handle the authentication.
        """
        raise NotImplementedError(self.auth_match)

    def build_auth_header(self, auth, request):
        """Build the value of the header used to authenticate.

        :param auth: The auth parameters needed to build the header.
        :param request: The request needing authentication.

        :return: None or header.
        """
        raise NotImplementedError(self.build_auth_header)

    def auth_successful(self, request, response):
        """The authentification was successful for the request.

        Additional infos may be available in the response.

        :param request: The succesfully authenticated request.
        :param response: The server response (may contain auth info).
        """
        self._retry_count = None

    def get_user_password(self, auth):
        """Ask user for a password if none is already available.

        :param auth: authentication info gathered so far (from the initial url
            and then during dialog with the server).
        """
        auth_conf = config.AuthenticationConfig()
        user = auth.get('user', None)
        password = auth.get('password', None)
        realm = auth['realm']
        port = auth.get('port', None)
        if user is None:
            user = auth_conf.get_user(auth['protocol'], auth['host'], port=port, path=auth['path'], realm=realm, ask=True, prompt=self.build_username_prompt(auth))
        if user is not None and password is None:
            password = auth_conf.get_password(auth['protocol'], auth['host'], user, port=port, path=auth['path'], realm=realm, prompt=self.build_password_prompt(auth))
        return (user, password)

    def _build_password_prompt(self, auth):
        """Build a prompt taking the protocol used into account.

        The AuthHandler is used by http and https, we want that information in
        the prompt, so we build the prompt from the authentication dict which
        contains all the needed parts.

        Also, http and proxy AuthHandlers present different prompts to the
        user. The daughter classes should implements a public
        build_password_prompt using this method.
        """
        prompt = '%s' % auth['protocol'].upper() + ' %(user)s@%(host)s'
        realm = auth['realm']
        if realm is not None:
            prompt += ", Realm: '%s'" % realm
        prompt += ' password'
        return prompt

    def _build_username_prompt(self, auth):
        """Build a prompt taking the protocol used into account.

        The AuthHandler is used by http and https, we want that information in
        the prompt, so we build the prompt from the authentication dict which
        contains all the needed parts.

        Also, http and proxy AuthHandlers present different prompts to the
        user. The daughter classes should implements a public
        build_username_prompt using this method.
        """
        prompt = '%s' % auth['protocol'].upper() + ' %(host)s'
        realm = auth['realm']
        if realm is not None:
            prompt += ", Realm: '%s'" % realm
        prompt += ' username'
        return prompt

    def http_request(self, request):
        """Insert an authentication header if information is available"""
        auth = self.get_auth(request)
        if self.auth_params_reusable(auth):
            self.add_auth_header(request, self.build_auth_header(auth, request))
        return request
    https_request = http_request