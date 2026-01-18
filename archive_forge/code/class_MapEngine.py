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
class MapEngine(EngineFacet):
    """The abstract base class for different map operation implementations.

    :param execution_engine: the execution engine this sql engine will run on
    """

    @abstractmethod
    def map_dataframe(self, df: DataFrame, map_func: Callable[[PartitionCursor, LocalDataFrame], LocalDataFrame], output_schema: Any, partition_spec: PartitionSpec, on_init: Optional[Callable[[int, DataFrame], Any]]=None, map_func_format_hint: Optional[str]=None) -> DataFrame:
        """Apply a function to each partition after you partition the dataframe in a
        specified way.

        :param df: input dataframe
        :param map_func: the function to apply on every logical partition
        :param output_schema: |SchemaLikeObject| that can't be None.
          Please also understand :ref:`why we need this
          <tutorial:tutorials/beginner/interface:schema>`
        :param partition_spec: partition specification
        :param on_init: callback function when the physical partition is initializaing,
          defaults to None
        :param map_func_format_hint: the preferred data format for ``map_func``, it can
          be ``pandas``, `pyarrow`, etc, defaults to None. Certain engines can provide
          the most efficient map operations based on the hint.
        :return: the dataframe after the map operation

        .. note::

            Before implementing, you must read
            :ref:`this <tutorial:tutorials/advanced/execution_engine:map>`
            to understand what map is used for and how it should work.
        """
        raise NotImplementedError

    def map_bag(self, bag: Bag, map_func: Callable[[BagPartitionCursor, LocalBag], LocalBag], partition_spec: PartitionSpec, on_init: Optional[Callable[[int, Bag], Any]]=None) -> Bag:
        """Apply a function to each partition after you partition the bag in a
        specified way.

        :param df: input dataframe
        :param map_func: the function to apply on every logical partition
        :param partition_spec: partition specification
        :param on_init: callback function when the physical partition is initializaing,
          defaults to None
        :return: the bag after the map operation
        """
        raise NotImplementedError