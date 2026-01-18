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
class _WSGIRequestHandler(wsgiref.simple_server.WSGIRequestHandler):
    """Custom WSGIRequestHandler.

    Uses a named logger instead of printing to stderr.
    """

    def log_message(self, format, *args):
        _LOGGER.info(format, *args)