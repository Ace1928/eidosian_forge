import commands
import difflib
import getpass
import itertools
import os
import re
import subprocess
import sys
import tempfile
import types
from google.apputils import app
import gflags as flags
from google.apputils import shellutil
def _DiffTestOutput(stream, golden_filename):
    """Compare ouput of redirected stream to contents of golden file.

  Args:
    stream: Should be sys.stdout or sys.stderr.
    golden_filename: Absolute path to golden file.
  """
    assert _captured_streams.has_key(stream)
    cap = _captured_streams[stream]
    for cap_stream in _captured_streams.itervalues():
        cap_stream.StopCapture()
    try:
        _Diff(cap.filename(), golden_filename)
    finally:
        del _captured_streams[stream]
        for cap_stream in _captured_streams.itervalues():
            cap_stream.RestartCapture()