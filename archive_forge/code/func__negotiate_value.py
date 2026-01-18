import logging
import re
import sys
import warnings
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import UnsupportedAlgorithm
from requests.auth import AuthBase
from requests.models import Response
from requests.compat import urlparse, StringIO
from requests.structures import CaseInsensitiveDict
from requests.cookies import cookiejar_from_dict
from requests.packages.urllib3 import HTTPResponse
from .exceptions import MutualAuthenticationError, KerberosExchangeError
def _negotiate_value(response):
    """Extracts the gssapi authentication token from the appropriate header"""
    if hasattr(_negotiate_value, 'regex'):
        regex = _negotiate_value.regex
    else:
        regex = re.compile('(?:.*,)*\\s*Negotiate\\s*([^,]*),?', re.I)
        _negotiate_value.regex = regex
    authreq = response.headers.get('www-authenticate', None)
    if authreq:
        match_obj = regex.search(authreq)
        if match_obj:
            return match_obj.group(1)
    return None