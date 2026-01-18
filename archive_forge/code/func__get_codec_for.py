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
def _get_codec_for(source):
    for ext, codec_class in _CODECS.items():
        if source.endswith(ext):
            return codec_class
    return None