import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
def GetEnvironFallback(var_list, default):
    """Look up a key in the environment, with fallback to secondary keys
  and finally falling back to a default value."""
    for var in var_list:
        if var in os.environ:
            return os.environ[var]
    return default