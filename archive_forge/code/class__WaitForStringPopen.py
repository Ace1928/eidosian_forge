import importlib
import importlib.util
import inspect
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from PIL import Image
import pytest
import matplotlib as mpl
from matplotlib import _c_internal_utils
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.testing import subprocess_run_helper as _run_helper
class _WaitForStringPopen(subprocess.Popen):
    """
    A Popen that passes flags that allow triggering KeyboardInterrupt.
    """

    def __init__(self, *args, **kwargs):
        if sys.platform == 'win32':
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
        super().__init__(*args, **kwargs, env={**os.environ, 'MPLBACKEND': 'Agg', 'SOURCE_DATE_EPOCH': '0'}, stdout=subprocess.PIPE, universal_newlines=True)

    def wait_for(self, terminator):
        """Read until the terminator is reached."""
        buf = ''
        while True:
            c = self.stdout.read(1)
            if not c:
                raise RuntimeError(f'Subprocess died before emitting expected {terminator!r}')
            buf += c
            if buf.endswith(terminator):
                return