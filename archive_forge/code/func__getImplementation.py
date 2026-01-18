from __future__ import annotations
import errno
import gc
import io
import os
import signal
import stat
import sys
import traceback
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from zope.interface import implementer
from twisted.internet import abstract, error, fdesc
from twisted.internet._baseprocess import BaseProcess
from twisted.internet.interfaces import IProcessTransport
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
from twisted.python import failure, log
from twisted.python.runtime import platform
from twisted.python.util import switchUID
def _getImplementation(self):
    """
        Pick a method which gives correct results for C{_listOpenFDs} in this
        runtime environment.

        This involves a lot of very platform-specific checks, some of which may
        be relatively expensive.  Therefore the returned method should be saved
        and re-used, rather than always calling this method to determine what it
        is.

        See the implementation for the details of how a method is selected.
        """
    for impl in self._implementations:
        try:
            before = impl()
        except BaseException:
            continue
        with self.openfile('/dev/null', 'r'):
            after = impl()
        if before != after:
            return impl
    return impl