import asyncio
import socket
import sys
import dns._asyncbackend
import dns._features
import dns.exception
import dns.inet
class _NetworkBackend(_CoreAsyncNetworkBackend):

    def __init__(self, resolver, local_port, bootstrap_address, family):
        super().__init__()
        self._local_port = local_port
        self._resolver = resolver
        self._bootstrap_address = bootstrap_address
        self._family = family
        if local_port != 0:
            raise NotImplementedError('the asyncio transport for HTTPX cannot set the local port')

    async def connect_tcp(self, host, port, timeout, local_address, socket_options=None):
        addresses = []
        _, expiration = _compute_times(timeout)
        if dns.inet.is_address(host):
            addresses.append(host)
        elif self._bootstrap_address is not None:
            addresses.append(self._bootstrap_address)
        else:
            timeout = _remaining(expiration)
            family = self._family
            if local_address:
                family = dns.inet.af_for_address(local_address)
            answers = await self._resolver.resolve_name(host, family=family, lifetime=timeout)
            addresses = answers.addresses()
        for address in addresses:
            try:
                attempt_expiration = _expiration_for_this_attempt(2.0, expiration)
                timeout = _remaining(attempt_expiration)
                with anyio.fail_after(timeout):
                    stream = await anyio.connect_tcp(remote_host=address, remote_port=port, local_host=local_address)
                return _CoreAnyIOStream(stream)
            except Exception:
                pass
        raise httpcore.ConnectError

    async def connect_unix_socket(self, path, timeout, socket_options=None):
        raise NotImplementedError

    async def sleep(self, seconds):
        await anyio.sleep(seconds)