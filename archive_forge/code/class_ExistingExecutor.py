import collections
import fractions
import functools
import heapq
import inspect
import logging
import math
import random
import threading
from concurrent import futures
import futurist
from futurist import _utils as utils
class ExistingExecutor(ExecutorFactory):
    """An executor factory returning the existing object."""

    def __init__(self, executor, shutdown=False):
        self._executor = executor
        self.shutdown = shutdown

    def __call__(self):
        return self._executor