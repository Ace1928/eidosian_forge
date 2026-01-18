import ssl
import typing
import anyio
from .._exceptions import (
from .._utils import is_socket_readable
from .base import SOCKET_OPTION, AsyncNetworkBackend, AsyncNetworkStream
class AnyIOBackend(AsyncNetworkBackend):

    async def connect_tcp(self, host: str, port: int, timeout: typing.Optional[float]=None, local_address: typing.Optional[str]=None, socket_options: typing.Optional[typing.Iterable[SOCKET_OPTION]]=None) -> AsyncNetworkStream:
        if socket_options is None:
            socket_options = []
        exc_map = {TimeoutError: ConnectTimeout, OSError: ConnectError, anyio.BrokenResourceError: ConnectError}
        with map_exceptions(exc_map):
            with anyio.fail_after(timeout):
                stream: anyio.abc.ByteStream = await anyio.connect_tcp(remote_host=host, remote_port=port, local_host=local_address)
                for option in socket_options:
                    stream._raw_socket.setsockopt(*option)
        return AnyIOStream(stream)

    async def connect_unix_socket(self, path: str, timeout: typing.Optional[float]=None, socket_options: typing.Optional[typing.Iterable[SOCKET_OPTION]]=None) -> AsyncNetworkStream:
        if socket_options is None:
            socket_options = []
        exc_map = {TimeoutError: ConnectTimeout, OSError: ConnectError, anyio.BrokenResourceError: ConnectError}
        with map_exceptions(exc_map):
            with anyio.fail_after(timeout):
                stream: anyio.abc.ByteStream = await anyio.connect_unix(path)
                for option in socket_options:
                    stream._raw_socket.setsockopt(*option)
        return AnyIOStream(stream)

    async def sleep(self, seconds: float) -> None:
        await anyio.sleep(seconds)