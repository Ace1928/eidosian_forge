from __future__ import print_function
import logging
import socket
import sys
from six.moves import BaseHTTPServer
from six.moves import http_client
from six.moves import input
from six.moves import urllib
from oauth2client import _helpers
from oauth2client import client
class ClientRedirectServer(BaseHTTPServer.HTTPServer):
    """A server to handle OAuth 2.0 redirects back to localhost.

    Waits for a single request and parses the query parameters
    into query_params and then stops serving.
    """
    query_params = {}