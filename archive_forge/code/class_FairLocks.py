import contextlib
import errno
import functools
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import weakref
import fasteners
from oslo_config import cfg
from oslo_utils import reflection
from oslo_utils import timeutils
from oslo_concurrency._i18n import _
class FairLocks(object):
    """A garbage collected container of fair locks.

    With a fair lock, contending lockers will get the lock in the order in
    which they tried to acquire it.

    This collection internally uses a weak value dictionary so that when a
    lock is no longer in use (by any threads) it will automatically be
    removed from this container by the garbage collector.
    """

    def __init__(self):
        self._locks = weakref.WeakValueDictionary()
        self._lock = threading.Lock()

    def get(self, name):
        """Gets (or creates) a lock with a given name.

        :param name: The lock name to get/create (used to associate
                     previously created names with the same lock).

        Returns an newly constructed lock (or an existing one if it was
        already created for the given name).
        """
        with self._lock:
            try:
                return self._locks[name]
            except KeyError:
                rwlock = ReaderWriterLock()
                self._locks[name] = rwlock
                return rwlock