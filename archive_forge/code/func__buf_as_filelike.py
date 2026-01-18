import io
import re
import functools
import inspect
import os
import sys
import numbers
import warnings
from pathlib import Path, PurePath
from typing import (
from ase.atoms import Atoms
from importlib import import_module
from ase.parallel import parallel_function, parallel_generator
def _buf_as_filelike(self, data: Union[str, bytes]) -> IO:
    encoding = self.encoding
    if encoding is None:
        encoding = 'utf-8'
    if self.isbinary:
        if isinstance(data, str):
            data = data.encode(encoding)
    elif isinstance(data, bytes):
        data = data.decode(encoding)
    return self._ioclass(data)