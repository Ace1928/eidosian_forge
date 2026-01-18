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
def _open_and_write_file():
    """Try to open the target file, raise error on failure"""
    fout = None
    try:
        try:
            fout = self._get_sftp().file(abspath, mode='wb')
            fout.set_pipelined(True)
            writer(fout)
        except (paramiko.SSHException, OSError) as e:
            self._translate_io_exception(e, abspath, ': unable to open')
        if mode is not None:
            self._get_sftp().chmod(abspath, mode)
    finally:
        if fout is not None:
            fout.close()