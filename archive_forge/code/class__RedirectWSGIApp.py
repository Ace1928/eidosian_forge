from base64 import urlsafe_b64encode
import hashlib
import json
import logging
from string import ascii_letters, digits
import webbrowser
import wsgiref.simple_server
import wsgiref.util
import google.auth.transport.requests
import google.oauth2.credentials
import google_auth_oauthlib.helpers
class _RedirectWSGIApp(object):
    """WSGI app to handle the authorization redirect.

    Stores the request URI and displays the given success message.
    """

    def __init__(self, success_message):
        """
        Args:
            success_message (str): The message to display in the web browser
                the authorization flow is complete.
        """
        self.last_request_uri = None
        self._success_message = success_message

    def __call__(self, environ, start_response):
        """WSGI Callable.

        Args:
            environ (Mapping[str, Any]): The WSGI environment.
            start_response (Callable[str, list]): The WSGI start_response
                callable.

        Returns:
            Iterable[bytes]: The response body.
        """
        start_response('200 OK', [('Content-type', 'text/plain; charset=utf-8')])
        self.last_request_uri = wsgiref.util.request_uri(environ)
        return [self._success_message.encode('utf-8')]