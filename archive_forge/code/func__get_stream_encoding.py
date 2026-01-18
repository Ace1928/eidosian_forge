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
def _get_stream_encoding(stream):
    encoding = config.GlobalStack().get('output_encoding')
    if encoding is None:
        encoding = getattr(stream, 'encoding', None)
    if encoding is None:
        encoding = osutils.get_terminal_encoding(trace=True)
    return encoding