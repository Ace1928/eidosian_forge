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
class _DependencyDict(ParamDict):

    def __init__(self, data: IndexedOrderedDict[str, _Dependency]):
        super().__init__()
        for k, v in data.items():
            super().__setitem__(k, v)
        self.set_readonly()

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(key).get()

    def items(self) -> Iterable[Tuple[str, Any]]:
        for k in self.keys():
            yield (k, self[k])

    def values(self) -> List[Any]:
        return [self[k] for k in self.keys()]