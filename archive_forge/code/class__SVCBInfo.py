import socket
import time
from urllib.parse import urlparse
import dns.asyncbackend
import dns.inet
import dns.name
import dns.nameserver
import dns.query
import dns.rdtypes.svcbbase
class _SVCBInfo:

    def __init__(self, bootstrap_address, port, hostname, nameservers):
        self.bootstrap_address = bootstrap_address
        self.port = port
        self.hostname = hostname
        self.nameservers = nameservers

    def ddr_check_certificate(self, cert):
        """Verify that the _SVCBInfo's address is in the cert's subjectAltName (SAN)"""
        for name, value in cert['subjectAltName']:
            if name == 'IP Address' and value == self.bootstrap_address:
                return True
        return False

    def make_tls_context(self):
        ssl = dns.query.ssl
        ctx = ssl.create_default_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        return ctx

    def ddr_tls_check_sync(self, lifetime):
        ctx = self.make_tls_context()
        expiration = time.time() + lifetime
        with socket.create_connection((self.bootstrap_address, self.port), lifetime) as s:
            with ctx.wrap_socket(s, server_hostname=self.hostname) as ts:
                ts.settimeout(dns.query._remaining(expiration))
                ts.do_handshake()
                cert = ts.getpeercert()
                return self.ddr_check_certificate(cert)

    async def ddr_tls_check_async(self, lifetime, backend=None):
        if backend is None:
            backend = dns.asyncbackend.get_default_backend()
        ctx = self.make_tls_context()
        expiration = time.time() + lifetime
        async with await backend.make_socket(dns.inet.af_for_address(self.bootstrap_address), socket.SOCK_STREAM, 0, None, (self.bootstrap_address, self.port), lifetime, ctx, self.hostname) as ts:
            cert = await ts.getpeercert(dns.query._remaining(expiration))
            return self.ddr_check_certificate(cert)