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
def _queue_renamed_event(self, src_event, src_path, dst_path, src_dirname, dst_dirname):
    cls = DirMovedEvent if src_event.is_directory else FileMovedEvent
    dst_path = self._encode_path(dst_path)
    self.queue_event(cls(src_path, dst_path))
    self.queue_event(DirModifiedEvent(src_dirname))
    self.queue_event(DirModifiedEvent(dst_dirname))