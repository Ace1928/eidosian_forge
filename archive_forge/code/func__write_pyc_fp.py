import ast
from collections import defaultdict
import errno
import functools
import importlib.abc
import importlib.machinery
import importlib.util
import io
import itertools
import marshal
import os
from pathlib import Path
from pathlib import PurePath
import struct
import sys
import tokenize
import types
from typing import Callable
from typing import Dict
from typing import IO
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from _pytest._io.saferepr import DEFAULT_REPR_MAX_SIZE
from _pytest._io.saferepr import saferepr
from _pytest._version import version
from _pytest.assertion import util
from _pytest.config import Config
from _pytest.main import Session
from _pytest.pathlib import absolutepath
from _pytest.pathlib import fnmatch_ex
from _pytest.stash import StashKey
from _pytest.assertion.util import format_explanation as _format_explanation  # noqa:F401, isort:skip
def _write_pyc_fp(fp: IO[bytes], source_stat: os.stat_result, co: types.CodeType) -> None:
    fp.write(importlib.util.MAGIC_NUMBER)
    flags = b'\x00\x00\x00\x00'
    fp.write(flags)
    mtime = int(source_stat.st_mtime) & 4294967295
    size = source_stat.st_size & 4294967295
    fp.write(struct.pack('<LL', mtime, size))
    fp.write(marshal.dumps(co))