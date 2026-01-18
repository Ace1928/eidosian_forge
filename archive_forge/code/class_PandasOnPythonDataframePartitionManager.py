from modin.core.dataframe.pandas.partitioning.partition_manager import (
from modin.core.execution.python.common import PythonWrapper
from .partition import PandasOnPythonDataframePartition
from .virtual_partition import (
class PandasOnPythonDataframePartitionManager(PandasDataframePartitionManager):
    """
    Class for managing partitions with pandas storage format and Python engine.

    Inherits all functionality from ``PandasDataframePartitionManager`` base class.
    """
    _partition_class = PandasOnPythonDataframePartition
    _column_partitions_class = PandasOnPythonDataframeColumnPartition
    _row_partition_class = PandasOnPythonDataframeRowPartition
    _execution_wrapper = PythonWrapper