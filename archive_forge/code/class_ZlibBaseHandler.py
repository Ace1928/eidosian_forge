import asyncio
import zlib
from concurrent.futures import Executor
from typing import Optional, cast
class ZlibBaseHandler:

    def __init__(self, mode: int, executor: Optional[Executor]=None, max_sync_chunk_size: Optional[int]=MAX_SYNC_CHUNK_SIZE):
        self._mode = mode
        self._executor = executor
        self._max_sync_chunk_size = max_sync_chunk_size