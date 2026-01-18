import concurrent.futures as cf
import logging
import sys
from abc import ABC, abstractmethod
from enum import Enum
from threading import Event, RLock
from traceback import StackSummary, extract_stack
from typing import (
from uuid import uuid4
from adagio.exceptions import AbortedError, SkippedError, WorkflowBug
from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec, WorkflowSpec
from six import reraise  # type: ignore
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.convert import to_instance
from triad.utils.hash import to_uuid
def _build_task(self, spec: TaskSpec) -> _Task:
    if isinstance(spec, WorkflowSpec):
        task: _Task = _Workflow(spec, self.ctx, self)
    else:
        task = _Task(spec, self.ctx, self)
    self._set_configs(task, spec)
    self._set_inputs(task, spec)
    if isinstance(task, _Workflow):
        task._init_tasks()
    return task