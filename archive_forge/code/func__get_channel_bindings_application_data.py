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
def _get_channel_bindings_application_data(response):
    """
    https://tools.ietf.org/html/rfc5929 4. The 'tls-server-end-point' Channel Binding Type

    Gets the application_data value for the 'tls-server-end-point' CBT Type.
    This is ultimately the SHA256 hash of the certificate of the HTTPS endpoint
    appended onto tls-server-end-point. This value is then passed along to the
    kerberos library to bind to the auth response. If the socket is not an SSL
    socket or the raw HTTP object is not a urllib3 HTTPResponse then None will
    be returned and the Kerberos auth will use GSS_C_NO_CHANNEL_BINDINGS

    :param response: The original 401 response from the server
    :return: byte string used on the application_data.value field on the CBT struct
    """
    application_data = None
    raw_response = response.raw
    if isinstance(raw_response, HTTPResponse):
        try:
            if sys.version_info > (3, 0):
                socket = raw_response._fp.fp.raw._sock
            else:
                socket = raw_response._fp.fp._sock
        except AttributeError:
            warnings.warn('Failed to get raw socket for CBT; has urllib3 impl changed', NoCertificateRetrievedWarning)
        else:
            try:
                server_certificate = socket.getpeercert(True)
            except AttributeError:
                pass
            else:
                certificate_hash = _get_certificate_hash(server_certificate)
                application_data = b'tls-server-end-point:' + certificate_hash
    else:
        warnings.warn('Requests is running with a non urllib3 backend, cannot retrieve server certificate for CBT', NoCertificateRetrievedWarning)
    return application_data