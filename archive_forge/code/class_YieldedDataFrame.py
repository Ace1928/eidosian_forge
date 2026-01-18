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
class YieldedDataFrame(Yielded):
    """Yielded dataframe from :class:`~fugue.workflow.workflow.FugueWorkflow`.
    Users shouldn't create this object directly.

    :param yid: unique id for determinism
    """

    def __init__(self, yid: str):
        super().__init__(yid)
        self._df: Any = None

    @property
    def is_set(self) -> bool:
        return self._df is not None

    def set_value(self, df: DataFrame) -> None:
        """Set the yielded dataframe after compute. Users should not
        call it.

        :param path: file path
        """
        self._df = df

    @property
    def result(self) -> DataFrame:
        """The yielded dataframe, it will be set after the parent
        workflow is computed
        """
        assert_or_throw(self.is_set, 'value is not set')
        return self._df