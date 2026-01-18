import os
import mimetypes
from typing import Generator
from libcloud.utils.py3 import b, next
def _optimized_chunked_generator(data: bytes, chunk_size: int) -> Generator[bytes, None, bytes]:
    chunk_start = 0
    while chunk_start + chunk_size < len(data):
        yield data[chunk_start:chunk_start + chunk_size]
        chunk_start += chunk_size
    data = data[chunk_start:]
    return data