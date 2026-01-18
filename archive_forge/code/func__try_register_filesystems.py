from __future__ import absolute_import, print_function, division
import logging
import sys
from contextlib import contextmanager
from petl.compat import PY3
from petl.io.sources import register_reader, register_writer, get_reader, get_writer
def _try_register_filesystems():
    try:
        import fsspec
    except ImportError:
        logger.debug('# Missing fsspec package. Install with: pip install fsspec')
    else:
        try:
            _register_filesystems()
        except Exception as ex:
            raise ImportError('# ERROR: failed to register fsspec filesystems', ex)