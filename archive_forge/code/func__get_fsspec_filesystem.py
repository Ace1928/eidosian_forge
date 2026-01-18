import dataclasses
import glob as py_glob
import io
import os
import os.path
import sys
import tempfile
from tensorboard.compat.tensorflow_stub import compat, errors
def _get_fsspec_filesystem(filename):
    """
    _get_fsspec_filesystem checks if the provided protocol is known to fsspec
    and if so returns the filesystem wrapper for it.
    """
    if not FSSPEC_ENABLED:
        return None
    segment = filename.partition(FSSpecFileSystem.CHAIN_SEPARATOR)[0]
    protocol = segment.partition(FSSpecFileSystem.SEPARATOR)[0]
    if fsspec.get_filesystem_class(protocol):
        return _FSSPEC_FILESYSTEM
    else:
        return None