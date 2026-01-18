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
def _get_sftp(self):
    """Ensures that a connection is established"""
    connection = self._get_connection()
    if connection is None:
        connection, credentials = self._create_connection()
        self._set_connection(connection, credentials)
    return connection