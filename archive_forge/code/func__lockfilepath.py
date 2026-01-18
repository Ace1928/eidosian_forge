import binascii
import os
import mmap
import sys
import time
import errno
from io import BytesIO
from smmap import (
import hashlib
from gitdb.const import (
def _lockfilepath(self):
    return '%s.lock' % self._filepath