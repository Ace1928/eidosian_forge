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
def _write_wrapper(self, *args, **kwargs):
    function = self._writefunc()
    if function is None:
        raise ValueError(f'Cannot write to {self.name}-format')
    return function(*args, **kwargs)