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
def _rename_and_overwrite(self, abs_from, abs_to):
    """Do a fancy rename on the remote server.

        Using the implementation provided by osutils.
        """
    try:
        sftp = self._get_sftp()
        fancy_rename(abs_from, abs_to, rename_func=sftp.rename, unlink_func=sftp.remove)
    except (OSError, paramiko.SSHException) as e:
        self._translate_io_exception(e, abs_from, ': unable to rename to %r' % abs_to)