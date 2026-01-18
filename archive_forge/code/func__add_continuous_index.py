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
def _add_continuous_index(df: dd.DataFrame) -> Tuple[dd.DataFrame, int]:

    def _get_info(df: pd.DataFrame, partition_info: Any) -> pd.DataFrame:
        return pd.DataFrame(dict(no=[partition_info['number']], ct=[len(df)]))
    pinfo = df.index.to_frame(name=df.index.name).map_partitions(_get_info, meta={'no': int, 'ct': int}).compute()
    counts = pinfo.sort_values('no').ct.cumsum().tolist()
    starts = [0] + counts[0:-1]

    def _add_index(df: pd.DataFrame, partition_info: Any) -> pd.DataFrame:
        return df.assign(**{_FUGUE_DASK_TEMP_IDX_COLUMN: np.arange(len(df)) + starts[partition_info['number']]})
    orig_schema = list(df.dtypes.to_dict().items())
    idf = df.map_partitions(_add_index, meta=orig_schema + [(_FUGUE_DASK_TEMP_IDX_COLUMN, int)])
    idf = idf.set_index(_FUGUE_DASK_TEMP_IDX_COLUMN, drop=True)
    return (idf, counts[-1])