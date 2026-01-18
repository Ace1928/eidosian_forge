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
def _on_progress_exit_all(self, progress_all_handle: MailboxProgressAll) -> None:
    probe_handles = []
    progress_handles = progress_all_handle.get_progress_handles()
    for progress_handle in progress_handles:
        probe_handles.extend(progress_handle.get_probe_handles())
    assert probe_handles
    if self._check_orphaned():
        self._stopped.set()
    poll_exit_responses: List[Optional[pb.PollExitResponse]] = []
    for probe_handle in probe_handles:
        result = probe_handle.get_probe_result()
        if result:
            poll_exit_responses.append(result.response.poll_exit_response)
    Run._footer_file_pusher_status_info(poll_exit_responses, printer=self._printer)