import contextlib
import os
import pathlib
import shutil
import stat
import sys
import zipfile
import {module}
def _copy_archive(archive, new_archive, interpreter=None):
    """Copy an application archive, modifying the shebang line."""
    with _maybe_open(archive, 'rb') as src:
        first_2 = src.read(2)
        if first_2 == b'#!':
            first_2 = b''
            src.readline()
        with _maybe_open(new_archive, 'wb') as dst:
            _write_file_prefix(dst, interpreter)
            dst.write(first_2)
            shutil.copyfileobj(src, dst)
    if interpreter and isinstance(new_archive, str):
        os.chmod(new_archive, os.stat(new_archive).st_mode | stat.S_IEXEC)