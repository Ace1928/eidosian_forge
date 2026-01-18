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
class SimpleSpec(WorkflowSpec):

    def __init__(self, inputs=[], outputs=[]):
        ip = [InputSpec(x, int, False) for x in inputs]
        op = [OutputSpec(x, int, False) for x in outputs]
        super().__init__(inputs=ip, outputs=op)
        self.cursor = None

    def add(self, name, func, *dep):
        ts = function_to_taskspec(func, lambda ds: [d['data_type'] is str for d in ds])
        dependency = {}
        if len(ts.inputs) > 0:
            if len(ts.inputs) == 1:
                if len(dep) == 0:
                    dep = [self.cursor.name]
            for f, t in zip(ts.inputs.keys(), dep):
                if t.startswith('*'):
                    t = t[1:]
                elif '.' not in t:
                    t = t + '.' + self.tasks[t].outputs.get_key_by_index(0)
                dependency[f] = t
        self.cursor = self.add_task(name, ts, dependency=dependency)