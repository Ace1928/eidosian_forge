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
def _unwrap_stream(stream):
    inner = getattr(stream, 'buffer', None)
    if inner is None:
        inner = getattr(stream, 'stream', stream)
    return inner