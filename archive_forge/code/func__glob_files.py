import contextlib
import io
import os
import pathlib
from mmap import mmap
from typing import Any, Container, List, Optional, Union
from .stat import stat_result
def _glob_files(self, basename):
    if isinstance(basename, str):
        basename = pathlib.Path(basename)
    files = basename.parent.glob(basename.name + '.*')
    return sorted(files)