import asyncio
import mimetypes
import os
import pathlib
from typing import (  # noqa
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import ETAG_ANY, ETag, must_be_empty_body
from .typedefs import LooseHeaders, PathLike
from .web_exceptions import (
from .web_response import StreamResponse
def _get_file_path_stat_and_gzip(self, check_for_gzipped_file: bool) -> Tuple[pathlib.Path, os.stat_result, bool]:
    """Return the file path, stat result, and gzip status.

        This method should be called from a thread executor
        since it calls os.stat which may block.
        """
    filepath = self._path
    if check_for_gzipped_file:
        gzip_path = filepath.with_name(filepath.name + '.gz')
        try:
            return (gzip_path, gzip_path.stat(), True)
        except OSError:
            pass
    return (filepath, filepath.stat(), False)