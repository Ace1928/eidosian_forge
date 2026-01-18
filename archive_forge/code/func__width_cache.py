from typing import Union
import pandas
import pyarrow as pa
from pandas._typing import AnyArrayLike
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from ..dataframe.utils import ColNameCodec, arrow_to_pandas
from ..db_worker import DbTable
@property
def _width_cache(self):
    """
        Number of columns.

        Returns
        -------
        int
        """
    if isinstance(self._data, pa.Table):
        return self._data.num_columns
    else:
        return self._data.shape[1]