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
def _GetDefaultTestTmpdir():
    tmpdir = os.environ.get('TEST_TMPDIR', '')
    if not tmpdir:
        tmpdir = os.path.join(tempfile.gettempdir(), 'google_apputils_basetest')
    return tmpdir