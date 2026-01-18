import bz2
import gzip
import lzma
import os
import shutil
import struct
import tarfile
import warnings
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type, Union
from .. import config
from ._filelock import FileLock
from .logging import get_logger
class ZstdExtractor(MagicNumberBaseExtractor):
    magic_numbers = [b'(\xb5/\xfd']

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        if not config.ZSTANDARD_AVAILABLE:
            raise ImportError('Please pip install zstandard')
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        with open(input_path, 'rb') as ifh, open(output_path, 'wb') as ofh:
            dctx.copy_stream(ifh, ofh)