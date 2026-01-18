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
def _GetDefaultTestRandomSeed():
    random_seed = 301
    value = os.environ.get('TEST_RANDOM_SEED', '')
    try:
        random_seed = int(value)
    except ValueError:
        pass
    return random_seed