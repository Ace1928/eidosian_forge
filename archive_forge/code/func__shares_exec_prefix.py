from __future__ import annotations
import logging # isort:skip
import os
import sys
from os.path import (
from tempfile import NamedTemporaryFile
def _shares_exec_prefix(basedir: str) -> bool:
    """ Whether a give base directory is on the system exex prefix

    """
    prefix: str | None = sys.exec_prefix
    return prefix is not None and basedir.startswith(prefix)