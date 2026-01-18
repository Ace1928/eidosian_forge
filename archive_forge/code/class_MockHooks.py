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
class MockHooks(WorkflowHooks):

    def __init__(self, wf_ctx: 'WorkflowContext'):
        super().__init__(wf_ctx)
        self.lock = RLock()
        self.res = OrderedDict()
        self.skipped = set()
        self.failed = set()

    def on_task_change(self, task: '_Task', old_state: '_State', new_state: '_State', e=None):
        if new_state == _State.FINISHED and len(task.outputs) == 1:
            with self.lock:
                self.res[task.name] = task.outputs['_0'].value
        if new_state == _State.SKIPPED:
            self.skipped.add(task.name)
        if new_state == _State.FAILED:
            self.failed.add(task.name)