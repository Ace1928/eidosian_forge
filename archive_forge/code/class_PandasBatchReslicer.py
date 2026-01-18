import math
from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar
import numpy as np
import pandas as pd
import pyarrow as pa
from triad.utils.convert import to_size
from .iter import slice_iterable
class PandasBatchReslicer(BatchReslicer[pd.DataFrame]):

    def get_rows_and_size(self, batch: pd.DataFrame) -> Tuple[int, int]:
        return (batch.shape[0], batch.memory_usage(deep=True).sum())

    def take(self, batch: pd.DataFrame, start: int, length: int) -> pd.DataFrame:
        if start == 0 and length == batch.shape[0]:
            return batch
        return batch.iloc[start:start + length]

    def concat(self, batches: List[pd.DataFrame]) -> pd.DataFrame:
        if len(batches) == 1:
            return batches[0]
        return pd.concat(batches)