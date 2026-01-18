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
def CaptureTestStderr(outfile=None):
    if not outfile:
        outfile = os.path.join(FLAGS.test_tmpdir, 'captured.err')
    _CaptureTestOutput(sys.stderr, outfile)