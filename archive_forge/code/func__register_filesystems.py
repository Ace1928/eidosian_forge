from __future__ import absolute_import, print_function, division
import logging
import sys
from contextlib import contextmanager
from petl.compat import PY3
from petl.io.sources import register_reader, register_writer, get_reader, get_writer
def _register_filesystems(only_available=False):
    """Register all known fsspec implementations as remote source."""
    from fsspec.registry import known_implementations, registry
    _register_filesystems_from(known_implementations, only_available)
    _register_filesystems_from(registry, only_available)