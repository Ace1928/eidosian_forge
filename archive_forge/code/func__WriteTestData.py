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
def _WriteTestData(data, filename):
    """Write data into file named filename."""
    fd = os.open(filename, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 384)
    os.write(fd, data)
    os.close(fd)