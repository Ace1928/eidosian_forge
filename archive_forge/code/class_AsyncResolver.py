import asyncio
import socket
from typing import Any, Dict, List, Optional, Type, Union
from .abc import AbstractResolver
from .helpers import get_running_loop
class AsyncResolver(AbstractResolver):
    """Use the `aiodns` package to make asynchronous DNS lookups"""

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop]=None, *args: Any, **kwargs: Any) -> None:
        if aiodns is None:
            raise RuntimeError('Resolver requires aiodns library')
        self._loop = get_running_loop(loop)
        self._resolver = aiodns.DNSResolver(*args, loop=loop, **kwargs)
        if not hasattr(self._resolver, 'gethostbyname'):
            self.resolve = self._resolve_with_query

    async def resolve(self, host: str, port: int=0, family: int=socket.AF_INET) -> List[Dict[str, Any]]:
        try:
            resp = await self._resolver.gethostbyname(host, family)
        except aiodns.error.DNSError as exc:
            msg = exc.args[1] if len(exc.args) >= 1 else 'DNS lookup failed'
            raise OSError(msg) from exc
        hosts = []
        for address in resp.addresses:
            hosts.append({'hostname': host, 'host': address, 'port': port, 'family': family, 'proto': 0, 'flags': socket.AI_NUMERICHOST | socket.AI_NUMERICSERV})
        if not hosts:
            raise OSError('DNS lookup failed')
        return hosts

    async def _resolve_with_query(self, host: str, port: int=0, family: int=socket.AF_INET) -> List[Dict[str, Any]]:
        if family == socket.AF_INET6:
            qtype = 'AAAA'
        else:
            qtype = 'A'
        try:
            resp = await self._resolver.query(host, qtype)
        except aiodns.error.DNSError as exc:
            msg = exc.args[1] if len(exc.args) >= 1 else 'DNS lookup failed'
            raise OSError(msg) from exc
        hosts = []
        for rr in resp:
            hosts.append({'hostname': host, 'host': rr.host, 'port': port, 'family': family, 'proto': 0, 'flags': socket.AI_NUMERICHOST})
        if not hosts:
            raise OSError('DNS lookup failed')
        return hosts

    async def close(self) -> None:
        self._resolver.cancel()