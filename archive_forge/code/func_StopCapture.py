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
def StopCapture(self):
    """Remove output redirection."""
    self._stream.flush()
    os.dup2(self._uncaptured_fd, self._fd)