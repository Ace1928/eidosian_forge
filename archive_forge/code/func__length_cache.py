from typing import Union
import pandas
import pyarrow as pa
from pandas._typing import AnyArrayLike
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from ..dataframe.utils import ColNameCodec, arrow_to_pandas
from ..db_worker import DbTable
@property
def _length_cache(self):
    """
        Number of rows.

        Returns
        -------
        int
        """
    return len(self._data)