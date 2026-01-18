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
class WorkflowResultCache(WorkflowContextMember, ABC):
    """Interface for cachine workflow task outputs. This cache is
    normally for cross execution retrieval.

    The implementation should be thread safe, and all methods should catch all
    exceptions and not raise.
    """

    def __init__(self, wf_ctx: 'WorkflowContext'):
        super().__init__(wf_ctx)

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set `key` with `value`

        :param key: uuid string
        :param value: any value
        """
        raise NotImplementedError

    @abstractmethod
    def skip(self, key: str) -> None:
        """Skip `key`

        :param key: uuid string
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, key: str) -> Tuple[bool, bool, Any]:
        """Try to get value for `key`

        :param key: uuid string
        :return: <hasvalue>, <skipped>, <value>
        """
        raise NotImplementedError