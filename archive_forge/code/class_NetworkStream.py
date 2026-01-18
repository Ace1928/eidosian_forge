import ssl
import time
import typing
class NetworkStream:

    def read(self, max_bytes: int, timeout: typing.Optional[float]=None) -> bytes:
        raise NotImplementedError()

    def write(self, buffer: bytes, timeout: typing.Optional[float]=None) -> None:
        raise NotImplementedError()

    def close(self) -> None:
        raise NotImplementedError()

    def start_tls(self, ssl_context: ssl.SSLContext, server_hostname: typing.Optional[str]=None, timeout: typing.Optional[float]=None) -> 'NetworkStream':
        raise NotImplementedError()

    def get_extra_info(self, info: str) -> typing.Any:
        return None