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
def _on_file_created(self, event: 'wd_events.FileCreatedEvent') -> None:
    logger.info('file/dir created: %s', event.src_path)
    if os.path.isdir(event.src_path):
        return None
    self._file_count += 1
    if self._file_count % 100 == 0:
        emitter = self.emitter
        if emitter:
            emitter._timeout = int(self._file_count / 100) + 1
    save_name = LogicalPath(os.path.relpath(event.src_path, self._dir))
    self._get_file_event_handler(event.src_path, save_name).on_modified()