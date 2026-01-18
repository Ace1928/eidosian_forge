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
def _queue_dirs_modified(self, dirs_modified, ref_snapshot, new_snapshot):
    """
        Queues events for directory modifications by scanning the directory
        for changes.

        A scan is a comparison between two snapshots of the same directory
        taken at two different times. This also determines whether files
        or directories were created, which updated the modified timestamp
        for the directory.
        """
    if dirs_modified:
        for dir_modified in dirs_modified:
            self.queue_event(DirModifiedEvent(dir_modified))
        diff_events = new_snapshot - ref_snapshot
        for file_created in diff_events.files_created:
            self.queue_event(FileCreatedEvent(file_created))
        for directory_created in diff_events.dirs_created:
            self.queue_event(DirCreatedEvent(directory_created))