from __future__ import annotations
import logging # isort:skip
import os
import sys
from os.path import (
from tempfile import NamedTemporaryFile
def _no_access(basedir: str) -> bool:
    """ Return True if the given base dir is not accessible or writeable

    """
    return not os.access(basedir, os.W_OK | os.X_OK)