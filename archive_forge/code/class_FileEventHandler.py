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
class FileEventHandler(abc.ABC):

    def __init__(self, file_path: PathStr, save_name: LogicalPath, file_pusher: 'FilePusher', *args: Any, **kwargs: Any) -> None:
        self.file_path = file_path
        self.save_name = LogicalPath(save_name)
        self._file_pusher = file_pusher
        self._last_sync: Optional[float] = None

    @property
    @abc.abstractmethod
    def policy(self) -> 'PolicyName':
        raise NotImplementedError

    @abc.abstractmethod
    def on_modified(self, force: bool=False) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def finish(self) -> None:
        raise NotImplementedError

    def on_renamed(self, new_path: PathStr, new_name: LogicalPath) -> None:
        self.file_path = new_path
        self.save_name = new_name
        self.on_modified()