import io
import json
import logging
import os
import select
import signal
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Set, Tuple
from torch.distributed.elastic.timer.api import TimerClient, TimerRequest
class FileTimerRequest(TimerRequest):
    """
    Data object representing a countdown timer acquisition and release
    that is used between the ``FileTimerClient`` and ``FileTimerServer``.
    A negative ``expiration_time`` should be interpreted as a "release"
    request.
    ``signal`` is the signal to reap the worker process from the server
    process.
    """
    __slots__ = ['version', 'worker_pid', 'scope_id', 'expiration_time', 'signal']

    def __init__(self, worker_pid: int, scope_id: str, expiration_time: float, signal: int=0) -> None:
        self.version = 1
        self.worker_pid = worker_pid
        self.scope_id = scope_id
        self.expiration_time = expiration_time
        self.signal = signal

    def __eq__(self, other) -> bool:
        if isinstance(other, FileTimerRequest):
            return self.version == other.version and self.worker_pid == other.worker_pid and (self.scope_id == other.scope_id) and (self.expiration_time == other.expiration_time) and (self.signal == other.signal)
        return False

    def to_json(self) -> str:
        return json.dumps({'version': self.version, 'pid': self.worker_pid, 'scope_id': self.scope_id, 'expiration_time': self.expiration_time, 'signal': self.signal})