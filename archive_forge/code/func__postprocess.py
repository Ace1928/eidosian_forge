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
def _postprocess(idf: dd.DataFrame, ct: int, num: int) -> dd.DataFrame:
    parts = min(ct, num)
    if parts <= 1:
        return idf.repartition(1)
    divisions = list(np.arange(ct, step=math.ceil(ct / parts)))
    divisions.append(ct - 1)
    return idf.repartition(divisions=divisions, force=True)