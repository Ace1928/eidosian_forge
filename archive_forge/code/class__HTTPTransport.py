import asyncio
import socket
import sys
import dns._asyncbackend
import dns._features
import dns.exception
import dns.inet
class _HTTPTransport(httpx.AsyncHTTPTransport):

    def __init__(self, *args, local_port=0, bootstrap_address=None, resolver=None, family=socket.AF_UNSPEC, **kwargs):
        if resolver is None:
            import dns.asyncresolver
            resolver = dns.asyncresolver.Resolver()
        super().__init__(*args, **kwargs)
        self._pool._network_backend = _NetworkBackend(resolver, local_port, bootstrap_address, family)