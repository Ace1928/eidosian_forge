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
class Interrupt(BaseException):

    def __init__(self, task, error):
        self.task = task
        self.error = error