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
def _process_add(self, action: StreamAction) -> None:
    stream = StreamRecord(action._data, mailbox=self._mailbox)
    settings = action._data
    thread = StreamThread(target=wandb.wandb_sdk.internal.internal.wandb_internal, kwargs=dict(settings=settings, record_q=stream._record_q, result_q=stream._result_q, port=self._port, user_pid=self._pid))
    stream.start_thread(thread)
    with self._streams_lock:
        self._streams[action._stream_id] = stream