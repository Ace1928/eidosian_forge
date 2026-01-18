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
def _on_file_moved(self, event: 'wd_events.FileMovedEvent') -> None:
    logger.info(f'file/dir moved: {event.src_path} -> {event.dest_path}')
    if os.path.isdir(event.dest_path):
        return None
    old_save_name = LogicalPath(os.path.relpath(event.src_path, self._dir))
    new_save_name = LogicalPath(os.path.relpath(event.dest_path, self._dir))
    handler = self._get_file_event_handler(event.src_path, old_save_name)
    self._file_event_handlers[new_save_name] = handler
    del self._file_event_handlers[old_save_name]
    handler.on_renamed(event.dest_path, new_save_name)