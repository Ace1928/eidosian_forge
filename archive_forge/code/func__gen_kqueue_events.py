from __future__ import annotations
import errno
import os
import os.path
import select
import threading
from stat import S_ISDIR
from watchdog.events import (
from watchdog.observers.api import DEFAULT_EMITTER_TIMEOUT, DEFAULT_OBSERVER_TIMEOUT, BaseObserver, EventEmitter
from watchdog.utils import platform
from watchdog.utils.dirsnapshot import DirectorySnapshot
def _gen_kqueue_events(self, kev, ref_snapshot, new_snapshot):
    """
        Generate events from the kevent list returned from the call to
        :meth:`select.kqueue.control`.

        .. NOTE:: kqueue only tells us about deletions, file modifications,
                  attribute modifications. The other events, namely,
                  file creation, directory modification, file rename,
                  directory rename, directory creation, etc. are
                  determined by comparing directory snapshots.
        """
    descriptor = self._descriptors.get_for_fd(kev.ident)
    src_path = descriptor.path
    if is_renamed(kev):
        for event in self._gen_renamed_events(src_path, descriptor.is_directory, ref_snapshot, new_snapshot):
            yield event
    elif is_attrib_modified(kev):
        if descriptor.is_directory:
            yield DirModifiedEvent(src_path)
        else:
            yield FileModifiedEvent(src_path)
    elif is_modified(kev):
        if descriptor.is_directory:
            if self.watch.is_recursive or self.watch.path == src_path:
                yield DirModifiedEvent(src_path)
        else:
            yield FileModifiedEvent(src_path)
    elif is_deleted(kev):
        if descriptor.is_directory:
            yield DirDeletedEvent(src_path)
        else:
            yield FileDeletedEvent(src_path)