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
class WorkflowContext(object):
    """Context of the workflow instance

    :param cache: cache type, instance or string representation,
      defaults to NoOpCache
    :param engine: engine type, instance or string representation,
      defaults to SequentialExecutionEngine
    :param hooks: hooks type, instance or string representation,
      defaults to WorkflowHooks
    :param logger: hooks type, instance or string representation,
      defaults to None (logging.getLogger())
    :param config: dict like configurations
    """

    @no_type_check
    def __init__(self, cache: Any=NoOpCache, engine: Any=SequentialExecutionEngine, hooks: Any=WorkflowHooks, logger: Any=None, config: Any=None):
        self._conf: ParamDict = ParamDict(config)
        self._abort_requested: Event = Event()
        self._cache: WorkflowResultCache = self._parse_config(cache, WorkflowResultCache, [self])
        self._engine: WorkflowExecutionEngine = self._parse_config(engine, WorkflowExecutionEngine, [self])
        self._hooks: WorkflowHooks = self._parse_config(hooks, WorkflowHooks, [self])
        if logger is None:
            logger = logging.getLogger()
        self._logger: logging.Logger = self._parse_config(logger, logging.Logger, [])

    @property
    def log(self) -> logging.Logger:
        """Logger for the workflow"""
        return self._logger

    @property
    def cache(self) -> WorkflowResultCache:
        """Cacher for the workflow"""
        return self._cache

    @property
    def conf(self) -> ParamDict:
        """Configs for the workflow"""
        return self._conf

    @property
    def hooks(self) -> WorkflowHooks:
        """Hooks for the workflow"""
        return self._hooks

    def abort(self) -> None:
        """Call this function to abort a running workflow"""
        self._abort_requested.set()

    @property
    def abort_requested(self) -> bool:
        """Abort requested"""
        return self._abort_requested.is_set()

    def run(self, spec: WorkflowSpec, conf: Dict[str, Any]) -> None:
        """Instantiate and run a workflow spec

        :param spec: workflow spec
        :param conf: configs to initialize the workflow
        """
        self._engine.run(spec, conf)

    def _parse_config(self, data: Any, tp: Type[WFMT], args: List[Any]) -> WFMT:
        if isinstance(data, tp):
            return data
        return cast(WFMT, to_instance(data, expected_base_type=tp, args=args))