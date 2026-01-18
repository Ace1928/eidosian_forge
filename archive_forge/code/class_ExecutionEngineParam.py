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
@fugue_annotated_param(ExecutionEngine, 'e', child_can_reuse_code=True)
class ExecutionEngineParam(AnnotatedParam):

    def __init__(self, param: Optional[inspect.Parameter]):
        super().__init__(param)
        self._type = self.annotation

    def to_input(self, engine: Any) -> Any:
        assert_or_throw(isinstance(engine, self._type), FugueWorkflowRuntimeError(f'{engine} is not of type {self._type}'))
        return engine

    def __uuid__(self) -> str:
        return to_uuid(self.code, self.annotation, self._type)