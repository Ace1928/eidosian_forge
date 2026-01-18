from modin.core.dataframe.pandas.partitioning.partition_manager import (
from modin.core.execution.dask.common import DaskWrapper
from .partition import PandasOnDaskDataframePartition
from .virtual_partition import (
class PandasOnDaskDataframePartitionManager(PandasDataframePartitionManager):
    """The class implements the interface in `PandasDataframePartitionManager`."""
    _partition_class = PandasOnDaskDataframePartition
    _column_partitions_class = PandasOnDaskDataframeColumnPartition
    _row_partition_class = PandasOnDaskDataframeRowPartition
    _execution_wrapper = DaskWrapper

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
        cls._execution_wrapper.wait([block for partition in partitions for block in partition.list_of_blocks])