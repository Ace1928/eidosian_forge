import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
def ExceptionAppend(e, msg):
    """Append a message to the given exception's message."""
    if not e.args:
        e.args = (msg,)
    elif len(e.args) == 1:
        e.args = (str(e.args[0]) + ' ' + msg,)
    else:
        e.args = (str(e.args[0]) + ' ' + msg,) + e.args[1:]