import ssl
import time
import typing
class AsyncNetworkBackend:

    async def connect_tcp(self, host: str, port: int, timeout: typing.Optional[float]=None, local_address: typing.Optional[str]=None, socket_options: typing.Optional[typing.Iterable[SOCKET_OPTION]]=None) -> AsyncNetworkStream:
        raise NotImplementedError()

    async def connect_unix_socket(self, path: str, timeout: typing.Optional[float]=None, socket_options: typing.Optional[typing.Iterable[SOCKET_OPTION]]=None) -> AsyncNetworkStream:
        raise NotImplementedError()

    async def sleep(self, seconds: float) -> None:
        raise NotImplementedError()