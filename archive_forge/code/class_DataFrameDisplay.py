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
class DataFrameDisplay(DatasetDisplay):
    """:class:`~.DataFrame` plain display class"""

    @property
    def df(self) -> DataFrame:
        """The target :class:`~.DataFrame`"""
        return self._ds

    def show(self, n: int=10, with_count: bool=False, title: Optional[str]=None) -> None:
        best_width = 100
        head_rows = self.df.head(n).as_array(type_safe=True)
        if len(head_rows) < n:
            count = len(head_rows)
        else:
            count = self.df.count() if with_count else -1
        with DatasetDisplay._SHOW_LOCK:
            if title is not None and title != '':
                print(title)
            print(type(self.df).__name__)
            tb = PrettyTable(self.df.schema, head_rows, best_width)
            print('\n'.join(tb.to_string()))
            if count >= 0:
                print(f'Total count: {count}')
                print('')
            if self.df.has_metadata:
                print('Metadata:')
                try:
                    print(self.df.metadata.to_json(indent=True))
                except Exception:
                    print(self.df.metadata)
                print('')