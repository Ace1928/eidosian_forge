import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
class MailboxProgress:
    _percent_done: float
    _handle: 'MailboxHandle'
    _probe_handles: List[MailboxProbe]
    _stopped: bool

    def __init__(self, _handle: 'MailboxHandle') -> None:
        self._handle = _handle
        self._percent_done = 0.0
        self._probe_handles = []
        self._stopped = False

    @property
    def percent_done(self) -> float:
        return self._percent_done

    def set_percent_done(self, percent_done: float) -> None:
        self._percent_done = percent_done

    def add_probe_handle(self, probe_handle: MailboxProbe) -> None:
        self._probe_handles.append(probe_handle)

    def get_probe_handles(self) -> List[MailboxProbe]:
        return self._probe_handles

    def wait_stop(self) -> None:
        self._stopped = True

    @property
    def _is_stopped(self) -> bool:
        return self._stopped