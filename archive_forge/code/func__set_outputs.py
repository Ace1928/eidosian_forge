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
def _set_outputs(self) -> None:
    assert isinstance(self.spec, WorkflowSpec)
    for f, to_expr in self.spec.internal_dependency.items():
        t = to_expr.split('.', 1)
        if len(t) == 1:
            self.outputs[f].set_dependency(self.inputs[t[0]])
        else:
            self.outputs[f].set_dependency(self.tasks[t[0]].outputs[t[1]])