import contextlib
import os
import pathlib
import shutil
import stat
import sys
import zipfile
import {module}
@contextlib.contextmanager
def _maybe_open(archive, mode):
    if isinstance(archive, (str, os.PathLike)):
        with open(archive, mode) as f:
            yield f
    else:
        yield archive