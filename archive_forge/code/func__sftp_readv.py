import bisect
import errno
import itertools
import os
import random
import stat
import sys
import time
import warnings
from .. import config, debug, errors, urlutils
from ..errors import LockError, ParamikoNotPresent, PathError, TransportError
from ..osutils import fancy_rename
from ..trace import mutter, warning
from ..transport import (ConnectedTransport, FileExists, FileFileStream,
def _sftp_readv(self, fp, offsets, relpath):
    """Use the readv() member of fp to do async readv.

        Then read them using paramiko.readv(). paramiko.readv()
        does not support ranges > 64K, so it caps the request size, and
        just reads until it gets all the stuff it wants.
        """
    helper = _SFTPReadvHelper(offsets, relpath, self._report_activity)
    return helper.request_and_yield_offsets(fp)