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
def _getchar(self):
    char = osutils.getchar()
    if char == chr(3):
        raise KeyboardInterrupt
    if char == chr(4):
        raise EOFError
    if isinstance(char, bytes):
        return char.decode('ascii', 'replace')
    return char