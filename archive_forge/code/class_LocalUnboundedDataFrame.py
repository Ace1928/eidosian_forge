import json
from abc import abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
import pandas as pd
import pyarrow as pa
from triad import SerializableRLock
from triad.collections.schema import Schema
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PD_UTILS
from triad.utils.pyarrow import cast_pa_table
from .._utils.display import PrettyTable
from ..collections.yielded import Yielded
from ..dataset import (
from ..exceptions import FugueDataFrameOperationError
class LocalUnboundedDataFrame(LocalDataFrame):
    """Base class of all local unbounded dataframes. Read
    this <https://fugue-tutorials.readthedocs.io/
    en/latest/tutorials/advanced/schema_dataframes.html#DataFrame>`_
    to understand the concept

    :param schema: |SchemaLikeObject|

    .. note::

        This is an abstract class, and normally you don't construct it by yourself
        unless you are
        implementing a new :class:`~fugue.execution.execution_engine.ExecutionEngine`
    """

    @property
    def is_bounded(self):
        """Always False because it's an unbounded dataframe"""
        return False

    def as_local(self) -> 'LocalDataFrame':
        return self

    def count(self) -> int:
        """
        :raises InvalidOperationError: You can't count an unbounded dataframe
        """
        raise InvalidOperationError('Impossible to count an LocalUnboundedDataFrame')