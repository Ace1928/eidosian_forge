from bisect import bisect_left, bisect_right
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Generator, List, Optional, Tuple
from zipfile import BadZipFile, ZipFile
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.requests.models import CONTENT_CHUNK_SIZE, Response
from pip._internal.metadata import BaseDistribution, MemoryWheel, get_wheel_distribution
from pip._internal.network.session import PipSession
from pip._internal.network.utils import HEADERS, raise_for_status, response_chunks
def _check_zip(self) -> None:
    """Check and download until the file is a valid ZIP."""
    end = self._length - 1
    for start in reversed(range(0, end, self._chunk_size)):
        self._download(start, end)
        with self._stay():
            try:
                ZipFile(self)
            except BadZipFile:
                pass
            else:
                break