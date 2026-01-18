import asyncio
import concurrent.futures
import errno
import os
import sys
import socket
import ssl
import stat
from tornado.concurrent import dummy_executor, run_on_executor
from tornado.ioloop import IOLoop
from tornado.util import Configurable, errno_from_exception
from typing import List, Callable, Any, Type, Dict, Union, Tuple, Awaitable, Optional
class DefaultLoopResolver(Resolver):
    """Resolver implementation using `asyncio.loop.getaddrinfo`."""

    async def resolve(self, host: str, port: int, family: socket.AddressFamily=socket.AF_UNSPEC) -> List[Tuple[int, Any]]:
        return [(fam, address) for fam, _, _, _, address in await asyncio.get_running_loop().getaddrinfo(host, port, family=family, type=socket.SOCK_STREAM)]