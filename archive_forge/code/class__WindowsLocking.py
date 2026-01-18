from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import enum
import errno
import hashlib
import io
import logging
import os
import shutil
import stat
import sys
import tempfile
import time
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding as encoding_util
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import retry
import six
from six.moves import range  # pylint: disable=redefined-builtin
class _WindowsLocking(object):
    """Exclusive, non-blocking file locking on Windows."""

    def TryLock(self, fd):
        """Raises IOError on failure."""
        import msvcrt
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def Unlock(self, fd):
        import msvcrt
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)