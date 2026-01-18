import abc
import fnmatch
import glob
import logging
import os
import queue
import time
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, MutableSet, Optional
from wandb import util
from wandb.sdk.interface.interface import GlobStr
from wandb.sdk.lib.paths import LogicalPath
class PolicyNow(FileEventHandler):
    """This policy only uploads files now."""

    def on_modified(self, force: bool=False) -> None:
        if self._last_sync is None or force:
            self._file_pusher.file_changed(self.save_name, self.file_path)
            self._last_sync = os.path.getmtime(self.file_path)

    def finish(self) -> None:
        pass

    @property
    def policy(self) -> 'PolicyName':
        return 'now'