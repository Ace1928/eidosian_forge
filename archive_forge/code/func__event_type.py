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