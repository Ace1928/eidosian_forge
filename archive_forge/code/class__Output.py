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
class _Output(_Dependency):

    def __init__(self, task: '_Task', spec: OutputSpec):
        super().__init__()
        self.task = task
        self.spec = spec
        self.exception: Optional[Exception] = None
        self.trace: Optional[StackSummary] = None
        self.value_set = Event()
        self.value: Any = None
        self.skipped = False
        self._lock = RLock()

    def __repr__(self) -> str:
        return f'{self.task}->{self.spec})'

    def __uuid__(self) -> str:
        return to_uuid(self.task, self.spec)

    def set(self, value: Any, from_cache: bool=False) -> '_Output':
        with self._lock:
            if not self.value_set.is_set():
                try:
                    self.value = self.spec.validate_value(value)
                    if self.task.spec.deterministic and (not from_cache):
                        self.task.ctx.cache.set(self.__uuid__(), self.value)
                    self.value_set.set()
                except Exception as e:
                    e = ValueError(str(e))
                    self.fail(e)
            return self

    def fail(self, exception: Exception, trace: Optional[StackSummary]=None, throw: bool=True) -> None:
        with self._lock:
            if not self.value_set.is_set():
                self.exception = exception
                self.trace = trace or extract_stack()
                self.value_set.set()
                if throw:
                    raise exception

    def skip(self, from_cache: bool=False) -> None:
        with self._lock:
            if not self.value_set.is_set():
                self.skipped = True
                if self.task.spec.deterministic and (not from_cache):
                    self.task.ctx.cache.skip(self.__uuid__())
                self.value_set.set()

    @property
    def is_set(self) -> bool:
        return self.value_set.is_set()

    @property
    def is_successful(self) -> bool:
        return self.value_set.is_set() and (not self.skipped) and (self.exception is None)

    @property
    def is_failed(self) -> bool:
        return self.value_set.is_set() and self.exception is not None

    @property
    def is_skipped(self) -> bool:
        return self.value_set.is_set() and self.skipped

    def validate_dependency(self, other: '_Dependency') -> None:
        aot(isinstance(other, (_Input, _Output)), lambda: TypeError(f'{other} is not Input or Output'))
        self.spec.validate_spec(other.spec)