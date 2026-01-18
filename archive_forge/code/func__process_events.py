import glob
import logging
import os
import queue
import socket
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import wandb
from wandb import util
from wandb.sdk.interface.interface import GlobStr
from wandb.sdk.lib import filesystem
from wandb.viz import CustomChart
from . import run as internal_run
def _process_events(self, shutdown_call: bool=False) -> None:
    try:
        with self._process_events_lock:
            for event in self._generator.Load():
                self.process_event(event)
    except (self.directory_watcher.DirectoryDeletedError, StopIteration, RuntimeError, OSError) as e:
        logger.debug('Encountered tensorboard directory watcher error: %s', e)
        if not self._shutdown.is_set() and (not shutdown_call):
            time.sleep(ERROR_DELAY)