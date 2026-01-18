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
def _setup_streams(self):
    self.raw_stdin = _unwrap_stream(self.stdin)
    self.stdin = _wrap_in_stream(self.raw_stdin)
    self.raw_stdout = _unwrap_stream(self.stdout)
    self.stdout = _wrap_out_stream(self.raw_stdout)
    self.raw_stderr = _unwrap_stream(self.stderr)
    self.stderr = _wrap_out_stream(self.raw_stderr)