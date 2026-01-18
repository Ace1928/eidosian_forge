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
def _write_pyc(state: 'AssertionState', co: types.CodeType, source_stat: os.stat_result, pyc: Path) -> bool:
    proc_pyc = f'{pyc}.{os.getpid()}'
    try:
        with open(proc_pyc, 'wb') as fp:
            _write_pyc_fp(fp, source_stat, co)
    except OSError as e:
        state.trace(f'error writing pyc file at {proc_pyc}: errno={e.errno}')
        return False
    try:
        os.replace(proc_pyc, pyc)
    except OSError as e:
        state.trace(f'error writing pyc file at {pyc}: {e}')
        return False
    return True