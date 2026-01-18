import asyncio
import zlib
from concurrent.futures import Executor
from typing import Optional, cast
class ZLibDecompressor(ZlibBaseHandler):

    def __init__(self, encoding: Optional[str]=None, suppress_deflate_header: bool=False, executor: Optional[Executor]=None, max_sync_chunk_size: Optional[int]=MAX_SYNC_CHUNK_SIZE):
        super().__init__(mode=encoding_to_mode(encoding, suppress_deflate_header), executor=executor, max_sync_chunk_size=max_sync_chunk_size)
        self._decompressor = zlib.decompressobj(wbits=self._mode)

    def decompress_sync(self, data: bytes, max_length: int=0) -> bytes:
        return self._decompressor.decompress(data, max_length)

    async def decompress(self, data: bytes, max_length: int=0) -> bytes:
        if self._max_sync_chunk_size is not None and len(data) > self._max_sync_chunk_size:
            return await asyncio.get_event_loop().run_in_executor(self._executor, self.decompress_sync, data, max_length)
        return self.decompress_sync(data, max_length)

    def flush(self, length: int=0) -> bytes:
        return self._decompressor.flush(length) if length > 0 else self._decompressor.flush()

    @property
    def eof(self) -> bool:
        return self._decompressor.eof

    @property
    def unconsumed_tail(self) -> bytes:
        return self._decompressor.unconsumed_tail

    @property
    def unused_data(self) -> bytes:
        return self._decompressor.unused_data