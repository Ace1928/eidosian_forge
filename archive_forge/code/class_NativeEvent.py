import os
import logging
import unicodedata
from threading import Thread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.events import (
from wandb_watchdog.observers.api import (
import AppKit
from FSEvents import (
from FSEvents import (
class NativeEvent(object):

    def __init__(self, path, flags, event_id):
        self.path = path
        self.flags = flags
        self.event_id = event_id
        self.is_created = bool(flags & kFSEventStreamEventFlagItemCreated)
        self.is_removed = bool(flags & kFSEventStreamEventFlagItemRemoved)
        self.is_renamed = bool(flags & kFSEventStreamEventFlagItemRenamed)
        self.is_modified = bool(flags & kFSEventStreamEventFlagItemModified)
        self.is_change_owner = bool(flags & kFSEventStreamEventFlagItemChangeOwner)
        self.is_inode_meta_mod = bool(flags & kFSEventStreamEventFlagItemInodeMetaMod)
        self.is_finder_info_mod = bool(flags & kFSEventStreamEventFlagItemFinderInfoMod)
        self.is_xattr_mod = bool(flags & kFSEventStreamEventFlagItemXattrMod)
        self.is_symlink = bool(flags & kFSEventStreamEventFlagItemIsSymlink)
        self.is_directory = bool(flags & kFSEventStreamEventFlagItemIsDir)

    @property
    def _event_type(self):
        if self.is_created:
            return 'Created'
        if self.is_removed:
            return 'Removed'
        if self.is_renamed:
            return 'Renamed'
        if self.is_modified:
            return 'Modified'
        if self.is_inode_meta_mod:
            return 'InodeMetaMod'
        if self.is_xattr_mod:
            return 'XattrMod'
        return 'Unknown'

    def __repr__(self):
        s = '<NativeEvent: path=%s, type=%s, is_dir=%s, flags=%s, id=%s>'
        return s % (repr(self.path), self._event_type, self.is_directory, hex(self.flags), self.event_id)