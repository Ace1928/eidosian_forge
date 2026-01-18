from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from xgboost import DataIter, DMatrix, QuantileDMatrix, XGBModel
from xgboost.compat import concat
from .._typing import ArrayLike
from .utils import get_logger  # type: ignore
class PartIter(DataIter):
    """Iterator for creating Quantile DMatrix from partitions."""

    def __init__(self, data: Dict[str, List], device_id: Optional[int], **kwargs: Any) -> None:
        self._iter = 0
        self._device_id = device_id
        self._data = data
        self._kwargs = kwargs
        super().__init__()

    def _fetch(self, data: Optional[Sequence[pd.DataFrame]]) -> Optional[pd.DataFrame]:
        if not data:
            return None
        if self._device_id is not None:
            import cudf
            import cupy as cp
            cp.cuda.runtime.setDevice(self._device_id)
            return cudf.DataFrame(data[self._iter])
        return data[self._iter]

    def next(self, input_data: Callable) -> int:
        if self._iter == len(self._data[alias.data]):
            return 0
        input_data(data=self._fetch(self._data[alias.data]), label=self._fetch(self._data.get(alias.label, None)), weight=self._fetch(self._data.get(alias.weight, None)), base_margin=self._fetch(self._data.get(alias.margin, None)), qid=self._fetch(self._data.get(alias.qid, None)), **self._kwargs)
        self._iter += 1
        return 1

    def reset(self) -> None:
        self._iter = 0