import numpy as np
from pandas.core.dtypes.common import is_numeric_dtype
from modin.config import AsyncReadMode
from modin.core.execution.modin_aqp import progress_bar_wrapper
from modin.core.execution.ray.common import RayWrapper
from modin.core.execution.ray.generic.partitioning import (
from modin.logging import get_logger
from modin.utils import _inherit_docstrings
from .partition import PandasOnRayDataframePartition
from .virtual_partition import (
class PandasOnRayDataframePartitionManager(GenericRayDataframePartitionManager):
    """The class implements the interface in `PandasDataframePartitionManager`."""
    _partition_class = PandasOnRayDataframePartition
    _column_partitions_class = PandasOnRayDataframeColumnPartition
    _row_partition_class = PandasOnRayDataframeRowPartition
    _execution_wrapper = RayWrapper
    materialize_futures = RayWrapper.materialize

    @classmethod
    def wait_partitions(cls, partitions):
        """
        Wait on the objects wrapped by `partitions` in parallel, without materializing them.

        This method will block until all computations in the list have completed.

        Parameters
        ----------
        partitions : np.ndarray
            NumPy array with ``PandasDataframePartition``-s.
        """
        RayWrapper.wait([block for partition in partitions for block in partition.list_of_blocks])

    @classmethod
    @_inherit_docstrings(GenericRayDataframePartitionManager.split_pandas_df_into_partitions)
    def split_pandas_df_into_partitions(cls, df, row_chunksize, col_chunksize, update_bar):
        enough_elements = len(df) * len(df.columns) > 6000000
        all_numeric_types = all((is_numeric_dtype(dtype) for dtype in df.dtypes))
        async_mode_on = AsyncReadMode.get()
        distributed_splitting = enough_elements and all_numeric_types and async_mode_on
        log = get_logger()
        if not distributed_splitting:
            log.info("Using sequential splitting in '.from_pandas()' because of some of the conditions are False: " + f'enough_elements={enough_elements!r}; all_numeric_types={all_numeric_types!r}; async_mode_on={async_mode_on!r}')
            return super().split_pandas_df_into_partitions(df, row_chunksize, col_chunksize, update_bar)
        log.info("Using distributed splitting in '.from_pandas()'")
        put_func = cls._partition_class.put

        def mask(part, row_loc, col_loc):
            return part.apply(lambda df: df.iloc[row_loc, :].iloc[:, col_loc])
        main_part = put_func(df)
        parts = [[update_bar(mask(main_part, slice(i, i + row_chunksize), slice(j, j + col_chunksize))) for j in range(0, len(df.columns), col_chunksize)] for i in range(0, len(df), row_chunksize)]
        return np.array(parts)