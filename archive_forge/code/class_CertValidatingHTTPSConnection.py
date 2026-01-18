import re
import socket
import ssl
import boto
from boto.compat import six, http_client
class CertValidatingHTTPSConnection(http_client.HTTPConnection):
    """An HTTPConnection that connects over SSL and validates certificates."""
    default_port = http_client.HTTPS_PORT

    def __init__(self, host, port=default_port, key_file=None, cert_file=None, ca_certs=None, strict=None, **kwargs):
        """Constructor.

        Args:
          host: The hostname. Can be in 'host:port' form.
          port: The port. Defaults to 443.
          key_file: A file containing the client's private key
          cert_file: A file containing the client's certificates
          ca_certs: A file contianing a set of concatenated certificate authority
              certs for validating the server against.
          strict: When true, causes BadStatusLine to be raised if the status line
              can't be parsed as a valid HTTP/1.0 or 1.1 status line.
        """
        if six.PY2:
            kwargs['strict'] = strict
        http_client.HTTPConnection.__init__(self, host=host, port=port, **kwargs)
        self.key_file = key_file
        self.cert_file = cert_file
        self.ca_certs = ca_certs

    def connect(self):
        """Connect to a host on a given (SSL) port."""
        if hasattr(self, 'timeout'):
            sock = socket.create_connection((self.host, self.port), self.timeout)
        else:
            sock = socket.create_connection((self.host, self.port))
        msg = 'wrapping ssl socket; '
        if self.ca_certs:
            msg += 'CA certificate file=%s' % self.ca_certs
        else:
            msg += 'using system provided SSL certs'
        boto.log.debug(msg)
        if hasattr(ssl, 'SSLContext') and getattr(ssl, 'HAS_SNI', False):
            context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
            context.verify_mode = ssl.CERT_REQUIRED
            if self.ca_certs:
                context.load_verify_locations(self.ca_certs)
            if self.cert_file:
                context.load_cert_chain(self.cert_file, self.key_file)
            self.sock = context.wrap_socket(sock, server_hostname=self.host)
            self.sock.keyfile = self.key_file
            self.sock.certfile = self.cert_file
            self.sock.cert_reqs = context.verify_mode
            self.sock.ssl_version = ssl.PROTOCOL_SSLv23
            self.sock.ca_certs = self.ca_certs
            self.sock.ciphers = None
        else:
            self.sock = ssl.wrap_socket(sock, keyfile=self.key_file, certfile=self.cert_file, cert_reqs=ssl.CERT_REQUIRED, ca_certs=self.ca_certs)
        cert = self.sock.getpeercert()
        hostname = self.host.split(':', 0)[0]
        if not ValidateCertificateHostname(cert, hostname):
            raise InvalidCertificateException(hostname, cert, 'remote hostname "%s" does not match certificate' % hostname)