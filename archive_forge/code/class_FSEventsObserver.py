from __future__ import with_statement
import sys
import threading
import unicodedata
import _watchdog_fsevents as _fsevents
from wandb_watchdog.events import (
from wandb_watchdog.utils.dirsnapshot import DirectorySnapshot
from wandb_watchdog.observers.api import (
class FSEventsObserver(BaseObserver):

    def __init__(self, timeout=DEFAULT_OBSERVER_TIMEOUT):
        BaseObserver.__init__(self, emitter_class=FSEventsEmitter, timeout=timeout)

    def schedule(self, event_handler, path, recursive=False):
        try:
            str_class = unicode
        except NameError:
            str_class = str
        if isinstance(path, str_class):
            path = unicodedata.normalize('NFC', path)
            if sys.version_info < (3,):
                path = path.encode('utf-8')
        return BaseObserver.schedule(self, event_handler, path, recursive)