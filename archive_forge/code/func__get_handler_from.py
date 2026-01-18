from __future__ import absolute_import, print_function, division
import os
import io
import gzip
import sys
import bz2
import zipfile
from contextlib import contextmanager
import subprocess
import logging
from petl.errors import ArgumentError
from petl.compat import urlopen, StringIO, BytesIO, string_types, PY2
def _get_handler_from(source, handlers):
    protocol_index = source.find('://')
    if protocol_index <= 0:
        return None
    protocol = source[:protocol_index]
    for prefix, handler_class in handlers.items():
        if prefix == protocol:
            return handler_class
    return None