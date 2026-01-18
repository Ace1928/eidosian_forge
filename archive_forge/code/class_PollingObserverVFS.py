from __future__ import with_statement
import os
import threading
from functools import partial
from wandb_watchdog.utils import stat as default_stat
from wandb_watchdog.utils.dirsnapshot import DirectorySnapshot, DirectorySnapshotDiff
from wandb_watchdog.observers.api import (
from wandb_watchdog.events import (
class PollingObserverVFS(BaseObserver):
    """
    File system independent observer that polls a directory to detect changes.
    """

    def __init__(self, stat, listdir, polling_interval=1):
        """
        :param stat: stat function. See ``os.stat`` for details.
        :param listdir: listdir function. See ``os.listdir`` for details.
        :type polling_interval: float
        :param polling_interval: interval in seconds between polling the file system.
        """
        emitter_cls = partial(PollingEmitter, stat=stat, listdir=listdir)
        BaseObserver.__init__(self, emitter_class=emitter_cls, timeout=polling_interval)