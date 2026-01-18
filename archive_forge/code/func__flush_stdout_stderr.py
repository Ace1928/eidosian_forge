from both of those two places to another location.
import errno
import logging
import os
import sys
import time
from io import StringIO
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import errors
def _flush_stdout_stderr():
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except ValueError:
        pass
    except OSError as e:
        import errno
        if e.errno in [errno.EINVAL, errno.EPIPE]:
            pass
        else:
            raise