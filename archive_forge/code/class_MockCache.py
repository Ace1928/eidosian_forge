import threading
import time
from collections import OrderedDict
from threading import RLock
from time import sleep
from typing import Any, Tuple
from adagio.exceptions import AbortedError
from adagio.instances import (NoOpCache, ParallelExecutionEngine, TaskContext,
from adagio.shells.interfaceless import function_to_taskspec
from adagio.specs import InputSpec, OutputSpec, WorkflowSpec, _NodeSpec
from pytest import raises
from triad.exceptions import InvalidOperationError
from timeit import timeit
class MockCache(WorkflowResultCache):

    def __init__(self, ctx=None):
        self.tb = dict()
        self.set_called = 0
        self.skip_called = 0
        self.get_called = 0
        self.hit = 0

    def set(self, key: str, value: Any) -> None:
        self.tb[key] = (False, value)
        print('set', key)
        self.set_called += 1

    def skip(self, key: str) -> None:
        self.tb[key] = (True, None)
        self.skip_called += 1

    def get(self, key: str) -> Tuple[bool, bool, Any]:
        self.get_called += 1
        if key not in self.tb:
            print('not get', key)
            return (False, False, None)
        x = self.tb[key]
        print('get', key)
        self.hit += 1
        return (True, x[0], x[1])