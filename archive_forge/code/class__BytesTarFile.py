from __future__ import annotations
from abc import (
import codecs
from collections import defaultdict
from collections.abc import (
import dataclasses
import functools
import gzip
from io import (
import mmap
import os
from pathlib import Path
import re
import tarfile
from typing import (
from urllib.parse import (
import warnings
import zipfile
from pandas._typing import (
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import ABCMultiIndex
from pandas.core.shared_docs import _shared_docs
class _BytesTarFile(_BufferedWriter):

    def __init__(self, name: str | None=None, mode: Literal['r', 'a', 'w', 'x']='r', fileobj: ReadBuffer[bytes] | WriteBuffer[bytes] | None=None, archive_name: str | None=None, **kwargs) -> None:
        super().__init__()
        self.archive_name = archive_name
        self.name = name
        self.buffer: tarfile.TarFile = tarfile.TarFile.open(name=name, mode=self.extend_mode(mode), fileobj=fileobj, **kwargs)

    def extend_mode(self, mode: str) -> str:
        mode = mode.replace('b', '')
        if mode != 'w':
            return mode
        if self.name is not None:
            suffix = Path(self.name).suffix
            if suffix in ('.gz', '.xz', '.bz2'):
                mode = f'{mode}:{suffix[1:]}'
        return mode

    def infer_filename(self) -> str | None:
        """
        If an explicit archive_name is not given, we still want the file inside the zip
        file not to be named something.tar, because that causes confusion (GH39465).
        """
        if self.name is None:
            return None
        filename = Path(self.name)
        if filename.suffix == '.tar':
            return filename.with_suffix('').name
        elif filename.suffix in ('.tar.gz', '.tar.bz2', '.tar.xz'):
            return filename.with_suffix('').with_suffix('').name
        return filename.name

    def write_to_buffer(self) -> None:
        archive_name = self.archive_name or self.infer_filename() or 'tar'
        tarinfo = tarfile.TarInfo(name=archive_name)
        tarinfo.size = len(self.getvalue())
        self.buffer.addfile(tarinfo, self)