from __future__ import print_function
import errno
import logging
import os
import time
from oauth2client import util
def _posix_lockfile(self, filename):
    """The name of the lock file to use for posix locking."""
    return '{0}.lock'.format(filename)