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
class _ConfigVar(_Dependency):

    def __init__(self, task: '_Task', spec: ConfigSpec):
        super().__init__()
        self.task = task
        self.is_set = False
        self.value: Any = None
        self.spec = spec

    def __repr__(self) -> str:
        return f'{self.spec}: {self.value}'

    def __uuid__(self) -> str:
        return to_uuid(self.get(), self.spec)

    def set(self, value: Any):
        self.value = self.spec.validate_value(value)
        self.is_set = True

    def get(self) -> Any:
        if self.dependency is not None:
            return self.dependency.get()
        if not self.is_set:
            aot(not self.spec.required, lambda: f'{self} is required but not set')
            return self.spec.default_value
        return self.value

    def validate_dependency(self, other: '_Dependency') -> None:
        aot(isinstance(other, _ConfigVar), lambda: TypeError(f'{other} is not Input or Output'))
        self.spec.validate_spec(other.spec)