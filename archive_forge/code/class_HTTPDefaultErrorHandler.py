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
class HTTPDefaultErrorHandler(urllib.request.HTTPDefaultErrorHandler):
    """Translate common errors into Breezy Exceptions"""

    def http_error_default(self, req, fp, code, msg, hdrs):
        if code == 403:
            raise errors.TransportError('Server refuses to fulfill the request (403 Forbidden) for %s' % req.get_full_url())
        else:
            raise errors.UnexpectedHttpStatus(req.get_full_url(), code, 'Unable to handle http code: %s' % msg, headers=hdrs)