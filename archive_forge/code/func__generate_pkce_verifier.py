import abc
import base64
import hashlib
import os
import time
from urllib import parse as urlparse
import warnings
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import federation
def _generate_pkce_verifier(self):
    """Generate PKCE verifier string as defined in RFC 7636."""
    raw_bytes = 42
    _rand = os.urandom(raw_bytes)
    _rand_b64 = base64.urlsafe_b64encode(_rand).decode('ascii')
    code_verifier = _rand_b64.rstrip('=')
    return code_verifier