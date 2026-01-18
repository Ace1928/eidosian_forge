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
class _IOWrapper:

    def __init__(self, buffer: BaseBuffer) -> None:
        self.buffer = buffer

    def __getattr__(self, name: str):
        return getattr(self.buffer, name)

    def readable(self) -> bool:
        if hasattr(self.buffer, 'readable'):
            return self.buffer.readable()
        return True

    def seekable(self) -> bool:
        if hasattr(self.buffer, 'seekable'):
            return self.buffer.seekable()
        return True

    def writable(self) -> bool:
        if hasattr(self.buffer, 'writable'):
            return self.buffer.writable()
        return True