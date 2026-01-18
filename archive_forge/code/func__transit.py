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
def _transit(self, new_state: _State, e: Optional[Exception]=None):
    with self._lock:
        old = self.state
        self.state = _State.transit(self.state, new_state)
        if e is not None:
            self._exception = e
            self._exec_info = sys.exc_info()
            self.log.error(f'{self} {old} -> {self.state}  {e}')
            self.ctx.hooks.on_task_change(self, old, self.state, e)
        else:
            self.log.debug(f'{self} {old} -> {self.state}')
            self.ctx.hooks.on_task_change(self, old, self.state)