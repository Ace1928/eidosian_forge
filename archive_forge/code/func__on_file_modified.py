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
def _on_file_modified(self, event: 'wd_events.FileModifiedEvent') -> None:
    logger.info(f'file/dir modified: {event.src_path}')
    if os.path.isdir(event.src_path):
        return None
    save_name = LogicalPath(os.path.relpath(event.src_path, self._dir))
    self._get_file_event_handler(event.src_path, save_name).on_modified()