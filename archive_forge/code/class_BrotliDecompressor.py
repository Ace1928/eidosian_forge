import asyncio
import zlib
from concurrent.futures import Executor
from typing import Optional, cast
class BrotliDecompressor:

    def __init__(self) -> None:
        if not HAS_BROTLI:
            raise RuntimeError('The brotli decompression is not available. Please install `Brotli` module')
        self._obj = brotli.Decompressor()

    def decompress_sync(self, data: bytes) -> bytes:
        if hasattr(self._obj, 'decompress'):
            return cast(bytes, self._obj.decompress(data))
        return cast(bytes, self._obj.process(data))

    def flush(self) -> bytes:
        if hasattr(self._obj, 'flush'):
            return cast(bytes, self._obj.flush())
        return b''