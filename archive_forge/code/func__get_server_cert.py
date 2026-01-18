import warnings
import base64
import typing as t
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import UnsupportedAlgorithm
from requests.auth import AuthBase
from requests.packages.urllib3.response import HTTPResponse
import spnego
def _get_server_cert(self, response):
    """
        Get the certificate at the request_url and return it as a hash. Will get the raw socket from the
        original response from the server. This socket is then checked if it is an SSL socket and then used to
        get the hash of the certificate. The certificate hash is then used with NTLMv2 authentication for
        Channel Binding Tokens support. If the raw object is not a urllib3 HTTPReponse (default with requests)
        then no certificate will be returned.

        :param response: The original 401 response from the server
        :return: The hash of the DER encoded certificate at the request_url or None if not a HTTPS endpoint
        """
    if self.send_cbt:
        raw_response = response.raw
        if isinstance(raw_response, HTTPResponse):
            socket = raw_response._fp.fp.raw._sock
            try:
                server_certificate = socket.getpeercert(True)
            except AttributeError:
                pass
            else:
                return _get_certificate_hash(server_certificate)
        else:
            warnings.warn('Requests is running with a non urllib3 backend, cannot retrieve server certificate for CBT', NoCertificateRetrievedWarning)
    return None