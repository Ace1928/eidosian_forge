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
def _post(self, body_bytes):
    """POST body_bytes to .bzr/smart on this transport.

        :returns: (response code, response body file-like object).
        """
    abspath = self._remote_path('.bzr/smart')
    response = self.request('POST', abspath, body=body_bytes, headers={'Content-Type': 'application/octet-stream'})
    code = response.status
    data = handle_response(abspath, code, response.getheader, response)
    return (code, data)