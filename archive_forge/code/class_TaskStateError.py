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
class TaskStateError(Exception):

    def __init__(self, state: TaskState, expected_state: TaskState) -> None:
        self.state = state
        self.expected_state = expected_state
        super().__init__(f'state: {state}, expected: {expected_state}')