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
def _QuoteLongString(s):
    """Quotes a potentially multi-line string to make the start and end obvious.

  Args:
    s: A string.

  Returns:
    The quoted string.
  """
    return '8<-----------\n' + s + '\n' + '----------->8\n'