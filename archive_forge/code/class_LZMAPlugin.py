from bz2 import BZ2File
from gzip import GzipFile
from io import IOBase
from lzma import LZMAFile
from typing import Any, BinaryIO, Dict, List
from scrapy.utils.misc import load_object
class LZMAPlugin:
    """
    Compresses received data using `lzma <https://en.wikipedia.org/wiki/Lempel–Ziv–Markov_chain_algorithm>`_.

    Accepted ``feed_options`` parameters:

    - `lzma_format`
    - `lzma_check`
    - `lzma_preset`
    - `lzma_filters`

    .. note::
        ``lzma_filters`` cannot be used in pypy version 7.3.1 and older.

    See :py:class:`lzma.LZMAFile` for more info about parameters.
    """

    def __init__(self, file: BinaryIO, feed_options: Dict[str, Any]) -> None:
        self.file = file
        self.feed_options = feed_options
        format = self.feed_options.get('lzma_format')
        check = self.feed_options.get('lzma_check', -1)
        preset = self.feed_options.get('lzma_preset')
        filters = self.feed_options.get('lzma_filters')
        self.lzmafile = LZMAFile(filename=self.file, mode='wb', format=format, check=check, preset=preset, filters=filters)

    def write(self, data: bytes) -> int:
        return self.lzmafile.write(data)

    def close(self) -> None:
        self.lzmafile.close()
        self.file.close()