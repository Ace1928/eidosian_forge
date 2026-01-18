from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
from ..partitioning.partition_manager import PandasOnPythonDataframePartitionManager
class PandasOnPythonDataframe(PandasDataframe):
    """
    Class for dataframes with pandas storage format and Python engine.

    ``PandasOnPythonDataframe`` doesn't implement any specific interfaces,
    all functionality is inherited from the ``PandasDataframe`` class.

    Parameters
    ----------
    partitions : np.ndarray
        A 2D NumPy array of partitions.
    index : sequence
        The index for the dataframe. Converted to a ``pandas.Index``.
    columns : sequence
        The columns object for the dataframe. Converted to a ``pandas.Index``.
    row_lengths : list, optional
        The length of each partition in the rows. The "height" of
        each of the block partitions. Is computed if not provided.
    column_widths : list, optional
        The width of each partition in the columns. The "width" of
        each of the block partitions. Is computed if not provided.
    dtypes : pandas.Series, optional
        The data types for the dataframe columns.
    """
    _partition_mgr_cls = PandasOnPythonDataframePartitionManager