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
@parallel_generator
def _iread(filename, index, format, io, parallel=None, full_output=False, **kwargs):
    if not io.can_read:
        raise ValueError("Can't read from {}-format".format(format))
    if io.single:
        start = index.start
        assert start is None or start == 0 or start == -1
        args = ()
    else:
        args = (index,)
    must_close_fd = False
    if isinstance(filename, str):
        if io.acceptsfd:
            mode = 'rb' if io.isbinary else 'r'
            fd = open_with_compression(filename, mode)
            must_close_fd = True
        else:
            fd = filename
    else:
        assert io.acceptsfd
        fd = filename
    try:
        for dct in io.read(fd, *args, **kwargs):
            if not isinstance(dct, dict):
                dct = {'atoms': dct}
            if full_output:
                yield dct
            else:
                yield dct['atoms']
    finally:
        if must_close_fd:
            fd.close()