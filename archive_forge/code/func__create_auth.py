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
def _create_auth(self):
    """Returns a dict containing the credentials provided at build time."""
    auth = dict(host=self._parsed_url.host, port=self._parsed_url.port, user=self._parsed_url.user, password=self._parsed_url.password, protocol=self._unqualified_scheme, path=self._parsed_url.path)
    return auth