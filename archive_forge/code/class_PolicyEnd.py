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
class PolicyEnd(FileEventHandler):
    """This policy only updates at the end of the run."""

    def on_modified(self, force: bool=False) -> None:
        pass

    def finish(self) -> None:
        self._last_sync = os.path.getmtime(self.file_path)
        self._file_pusher.file_changed(self.save_name, self.file_path, copy=False)

    @property
    def policy(self) -> 'PolicyName':
        return 'end'