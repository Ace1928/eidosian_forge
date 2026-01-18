import io
import socket
import ssl
from ..exceptions import ProxySchemeUnsupported
from ..packages import six
@staticmethod
def _validate_ssl_context_for_tls_in_tls(ssl_context):
    """
        Raises a ProxySchemeUnsupported if the provided ssl_context can't be used
        for TLS in TLS.

        The only requirement is that the ssl_context provides the 'wrap_bio'
        methods.
        """
    if not hasattr(ssl_context, 'wrap_bio'):
        if six.PY2:
            raise ProxySchemeUnsupported("TLS in TLS requires SSLContext.wrap_bio() which isn't supported on Python 2")
        else:
            raise ProxySchemeUnsupported("TLS in TLS requires SSLContext.wrap_bio() which isn't available on non-native SSLContext")