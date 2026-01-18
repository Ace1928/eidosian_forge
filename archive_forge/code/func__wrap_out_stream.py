import codecs
import io
import os
import sys
import warnings
from ..lazy_import import lazy_import
import time
from breezy import (
from .. import config, osutils, trace
from . import NullProgressView, UIFactory
def _wrap_out_stream(stream, encoding=None, errors='replace'):
    if encoding is None:
        encoding = _get_stream_encoding(stream)
    encoded_stream = codecs.getwriter(encoding)(stream, errors=errors)
    encoded_stream.encoding = encoding
    return encoded_stream