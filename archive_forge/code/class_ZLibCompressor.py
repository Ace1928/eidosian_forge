import asyncio
import zlib
from concurrent.futures import Executor
from typing import Optional, cast
class ZLibCompressor(ZlibBaseHandler):

    def __init__(self, encoding: Optional[str]=None, suppress_deflate_header: bool=False, level: Optional[int]=None, wbits: Optional[int]=None, strategy: int=zlib.Z_DEFAULT_STRATEGY, executor: Optional[Executor]=None, max_sync_chunk_size: Optional[int]=MAX_SYNC_CHUNK_SIZE):
        super().__init__(mode=encoding_to_mode(encoding, suppress_deflate_header) if wbits is None else wbits, executor=executor, max_sync_chunk_size=max_sync_chunk_size)
        if level is None:
            self._compressor = zlib.compressobj(wbits=self._mode, strategy=strategy)
        else:
            self._compressor = zlib.compressobj(wbits=self._mode, strategy=strategy, level=level)
        self._compress_lock = asyncio.Lock()

    def compress_sync(self, data: bytes) -> bytes:
        return self._compressor.compress(data)

    async def compress(self, data: bytes) -> bytes:
        async with self._compress_lock:
            if self._max_sync_chunk_size is not None and len(data) > self._max_sync_chunk_size:
                return await asyncio.get_event_loop().run_in_executor(self._executor, self.compress_sync, data)
            return self.compress_sync(data)

    def flush(self, mode: int=zlib.Z_FINISH) -> bytes:
        return self._compressor.flush(mode)