from __future__ import annotations
import logging
import os
import threading
import time
import unicodedata
import _watchdog_fsevents as _fsevents  # type: ignore[import-not-found]
from watchdog.events import (
from watchdog.observers.api import DEFAULT_EMITTER_TIMEOUT, DEFAULT_OBSERVER_TIMEOUT, BaseObserver, EventEmitter
from watchdog.utils.dirsnapshot import DirectorySnapshot
@staticmethod
def _is_meta_mod(event):
    """Returns True if the event indicates a change in metadata."""
    return event.is_inode_meta_mod or event.is_xattr_mod or event.is_owner_change