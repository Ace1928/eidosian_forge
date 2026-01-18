import math
from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar
import numpy as np
import pandas as pd
import pyarrow as pa
from triad.utils.convert import to_size
from .iter import slice_iterable
class ArrowTableSortedBatchReslicer(SortedBatchReslicer[pa.Table]):

    def get_keys_ndarray(self, batch: pa.Table, keys: List[str]) -> np.ndarray:
        return batch.select(keys).to_pandas().to_numpy()

    def get_batch_length(self, batch: pa.Table) -> int:
        return batch.num_rows

    def take(self, batch: pa.Table, start: int, length: int) -> pa.Table:
        return batch.slice(start, length)

    def concat(self, batches: List[pa.Table]) -> pa.Table:
        return pa.concat_tables(batches)