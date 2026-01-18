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
def _real_umount(self, mountpoint, rootwrap_helper):
    LOG.debug('Unmounting %(mountpoint)s', {'mountpoint': mountpoint})
    try:
        processutils.execute('umount', mountpoint, run_as_root=True, attempts=3, delay_on_retry=True, root_helper=rootwrap_helper)
    except processutils.ProcessExecutionError as ex:
        LOG.error(_LE("Couldn't unmount %(mountpoint)s: %(reason)s"), {'mountpoint': mountpoint, 'reason': ex})
    if not os.path.ismount(mountpoint):
        try:
            os.rmdir(mountpoint)
        except Exception as ex:
            LOG.error(_LE("Couldn't remove directory %(mountpoint)s: %(reason)s"), {'mountpoint': mountpoint, 'reason': ex})
        return False
    return True