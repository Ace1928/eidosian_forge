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
@dataclasses.dataclass
class IOArgs:
    """
    Return value of io/common.py:_get_filepath_or_buffer.
    """
    filepath_or_buffer: str | BaseBuffer
    encoding: str
    mode: str
    compression: CompressionDict
    should_close: bool = False