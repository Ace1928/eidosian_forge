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
def _parent_dir_modified(self, src_path):
    """
        Helper to generate a DirModifiedEvent on the parent of src_path.
        """
    return DirModifiedEvent(os.path.dirname(src_path))