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
def _process_teardown(self, action: StreamAction) -> None:
    exit_code: int = action._data
    with self._streams_lock:
        streams_copy = self._streams.copy()
    self._finish_all(streams_copy, exit_code)
    with self._streams_lock:
        self._streams = dict()
    self._stopped.set()