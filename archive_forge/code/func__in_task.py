import enum
import functools
import heapq
import itertools
import signal
import threading
import time
from concurrent.futures import Future
from contextvars import ContextVar
from typing import (
import duet.futuretools as futuretools
def _in_task(self, frame) -> bool:
    while frame is not None:
        if frame.f_locals.get(LOCALS_TASK_SCHEDULER, None) is self:
            return True
        frame = frame.f_back
    return False