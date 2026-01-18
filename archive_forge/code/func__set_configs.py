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
def _set_configs(self, task: _Task, spec: TaskSpec) -> None:
    for f, v in spec.node_spec.config.items():
        task.configs[f].set(v)
    for f, t in spec.node_spec.config_dependency.items():
        task.configs[f].set_dependency(self.configs[t])