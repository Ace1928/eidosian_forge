from __future__ import annotations
import os
from os.path import exists
from os.path import join
from os.path import splitext
from subprocess import check_call
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from .compat import is_posix
from .exc import CommandError

    Opens the given file in a text editor. If the environment variable
    ``EDITOR`` is set, this is taken as preference.

    Otherwise, a list of commonly installed editors is tried.

    If no editor matches, an :py:exc:`OSError` is raised.

    :param filename: The filename to open. Will be passed  verbatim to the
        editor command.
    :param environ: An optional drop-in replacement for ``os.environ``. Used
        mainly for testing.
    