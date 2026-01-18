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
def _assert_source_has_open(source_class):
    source = source_class('test')
    assert hasattr(source, 'open') and callable(getattr(source, 'open')), _invalid_source_msg % source