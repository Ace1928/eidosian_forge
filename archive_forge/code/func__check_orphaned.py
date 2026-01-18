import functools
import multiprocessing
import queue
import threading
import time
from threading import Event
from typing import Any, Callable, Dict, List, Optional
import psutil
import wandb
import wandb.util
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib.mailbox import (
from wandb.sdk.lib.printer import get_printer
from wandb.sdk.wandb_run import Run
from ..interface.interface_relay import InterfaceRelay
def _check_orphaned(self) -> bool:
    if not self._pid:
        return False
    time_now = time.time()
    if self._pid_checked_ts and time_now < self._pid_checked_ts + 2:
        return False
    self._pid_checked_ts = time_now
    return not psutil.pid_exists(self._pid)