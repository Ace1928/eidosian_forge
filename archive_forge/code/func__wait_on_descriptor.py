import _thread
import errno
import io
import os
import sys
import time
import breezy
from ...lazy_import import lazy_import
import select
import socket
import weakref
from breezy import (
from breezy.i18n import gettext
from breezy.bzr.smart import client, protocol, request, signals, vfs
from breezy.transport import ssh
from ... import errors, osutils
def _wait_on_descriptor(self, fd, timeout_seconds):
    """select() on a file descriptor, waiting for nonblocking read()

        This will raise a ConnectionTimeout exception if we do not get a
        readable handle before timeout_seconds.
        :return: None
        """
    t_end = self._timer() + timeout_seconds
    poll_timeout = min(timeout_seconds, self._client_poll_timeout)
    rs = xs = None
    while not rs and (not xs) and (self._timer() < t_end):
        if self.finished:
            return
        try:
            rs, _, xs = select.select([fd], [], [fd], poll_timeout)
        except OSError as e:
            err = getattr(e, 'errno', None)
            if err is None and getattr(e, 'args', None) is not None:
                err = e.args[0]
            if err in _bad_file_descriptor:
                return
            elif err == errno.EINTR:
                continue
            raise
        except ValueError:
            return
    if rs or xs:
        return
    raise errors.ConnectionTimeout('disconnecting client after %.1f seconds' % (timeout_seconds,))