import inspect
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
from uuid import uuid4
from triad import ParamDict, Schema, SerializableRLock, assert_or_throw, to_uuid
from triad.collections.function_wrapper import AnnotatedParam
from triad.exceptions import InvalidOperationError
from triad.utils.convert import to_size
from fugue.bag import Bag, LocalBag
from fugue.collections.partition import (
from fugue.collections.sql import StructuredRawSQL, TempTableName
from fugue.collections.yielded import PhysicalYielded, Yielded
from fugue.column import (
from fugue.constants import _FUGUE_GLOBAL_CONF, FUGUE_SQL_DEFAULT_DIALECT
from fugue.dataframe import AnyDataFrame, DataFrame, DataFrames, fugue_annotated_param
from fugue.dataframe.array_dataframe import ArrayDataFrame
from fugue.dataframe.dataframe import LocalDataFrame
from fugue.dataframe.utils import deserialize_df, serialize_df
from fugue.exceptions import FugueWorkflowRuntimeError
class _GlobalExecutionEngineContext:

    def __init__(self):
        self._engine: Optional['ExecutionEngine'] = None

    def set(self, engine: Optional['ExecutionEngine']):
        with _CONTEXT_LOCK:
            if self._engine is not None:
                self._engine._is_global = False
                self._engine._exit_context()
            self._engine = engine
            if engine is not None:
                engine._enter_context()
                engine._is_global = True

    def get(self) -> Optional['ExecutionEngine']:
        return self._engine