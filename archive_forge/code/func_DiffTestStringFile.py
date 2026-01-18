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
def DiffTestStringFile(data, golden):
    """Diff data agains a golden file."""
    data_file = os.path.join(FLAGS.test_tmpdir, 'provided.dat')
    _WriteTestData(data, data_file)
    _Diff(data_file, golden)