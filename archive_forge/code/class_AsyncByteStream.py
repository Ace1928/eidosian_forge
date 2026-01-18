import ssl
from http.cookiejar import CookieJar
from typing import (
class AsyncByteStream:

    async def __aiter__(self) -> AsyncIterator[bytes]:
        raise NotImplementedError("The '__aiter__' method must be implemented.")
        yield b''

    async def aclose(self) -> None:
        pass