import re
import socket
import ssl
import boto
from boto.compat import six, http_client
class InvalidCertificateException(http_client.HTTPException):
    """Raised when a certificate is provided with an invalid hostname."""

    def __init__(self, host, cert, reason):
        """Constructor.

        Args:
          host: The hostname the connection was made to.
          cert: The SSL certificate (as a dictionary) the host returned.
        """
        http_client.HTTPException.__init__(self)
        self.host = host
        self.cert = cert
        self.reason = reason

    def __str__(self):
        return 'Host %s returned an invalid certificate (%s): %s' % (self.host, self.reason, self.cert)