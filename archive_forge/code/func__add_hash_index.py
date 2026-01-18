import math
from typing import Any, Callable, List, Optional, Tuple, TypeVar
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
from dask.dataframe.core import DataFrame
from dask.delayed import delayed
from dask.distributed import Client, get_client
from triad.utils.pandas_like import PD_UTILS, PandasLikeUtils
from triad.utils.pyarrow import to_pandas_dtype
import fugue.api as fa
from fugue.constants import FUGUE_CONF_DEFAULT_PARTITIONS
from ._constants import FUGUE_DASK_CONF_DEFAULT_PARTITIONS
def _add_hash_index(df: dd.DataFrame, num: int, cols: List[str]) -> Tuple[dd.DataFrame, int]:
    if len(cols) == 0:
        cols = list(df.columns)

    def _add_hash(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return df.assign(**{_FUGUE_DASK_TEMP_IDX_COLUMN: pd.Series(dtype=int)})
        return df.assign(**{_FUGUE_DASK_TEMP_IDX_COLUMN: pd.util.hash_pandas_object(df[cols], index=False).mod(num).astype(int)})
    orig_schema = list(df.dtypes.to_dict().items())
    idf = df.map_partitions(_add_hash, meta=orig_schema + [(_FUGUE_DASK_TEMP_IDX_COLUMN, int)]).set_index(_FUGUE_DASK_TEMP_IDX_COLUMN, drop=True)
    return (idf, num)