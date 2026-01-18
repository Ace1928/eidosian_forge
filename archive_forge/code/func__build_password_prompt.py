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