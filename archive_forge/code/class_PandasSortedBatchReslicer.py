import math
from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar
import numpy as np
import pandas as pd
import pyarrow as pa
from triad.utils.convert import to_size
from .iter import slice_iterable
class PandasSortedBatchReslicer(SortedBatchReslicer[pd.DataFrame]):

    def get_keys_ndarray(self, batch: pd.DataFrame, keys: List[str]) -> np.ndarray:
        return batch[keys].to_numpy()

    def get_batch_length(self, batch: pd.DataFrame) -> int:
        return batch.shape[0]

    def take(self, batch: pd.DataFrame, start: int, length: int) -> pd.DataFrame:
        return batch.iloc[start:start + length]

    def concat(self, batches: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(batches)