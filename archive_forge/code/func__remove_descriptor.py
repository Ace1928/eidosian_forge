from __future__ import with_statement
from wandb_watchdog.utils import platform
import threading
import errno
import sys
import stat
import os
from wandb_watchdog.observers.api import (
from wandb_watchdog.utils.dirsnapshot import DirectorySnapshot
from wandb_watchdog.events import (
def _remove_descriptor(self, descriptor):
    """
        Removes a descriptor from the collection.

        :param descriptor:
            An instance of :class:`KeventDescriptor` to be removed.
        """
    self._descriptors.remove(descriptor)
    del self._descriptor_for_fd[descriptor.fd]
    del self._descriptor_for_path[descriptor.path]
    self._kevents.remove(descriptor.kevent)
    descriptor.close()