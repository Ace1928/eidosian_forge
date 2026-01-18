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
def _process_action(self, action: StreamAction) -> None:
    if action._action == 'add':
        self._process_add(action)
        return
    if action._action == 'update':
        self._process_update(action)
        return
    if action._action == 'start':
        self._process_start(action)
        return
    if action._action == 'del':
        self._process_del(action)
        return
    if action._action == 'drop':
        self._process_drop(action)
        return
    if action._action == 'teardown':
        self._process_teardown(action)
        return
    raise AssertionError(f'Unsupported action: {action._action}')