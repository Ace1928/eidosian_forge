import collections
import contextlib
import logging
import os
import socket
import threading
from oslo_concurrency import processutils
from oslo_config import cfg
from glance_store import exceptions
from glance_store.i18n import _LE, _LW
@contextlib.contextmanager
def _get_locked(self, mountpoint):
    """Get a locked mountpoint object

        :param mountpoint: The path of the mountpoint whose object we should
                           return.
        :rtype: _HostMountState._MountPoint
        """
    while True:
        mount = self.mountpoints[mountpoint]
        with mount.lock:
            if self.mountpoints[mountpoint] is mount:
                yield mount
                break