from modin.core.dataframe.base.dataframe.utils import Axis
from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
from ..partitioning.partition_manager import PandasOnRayDataframePartitionManager
class PandasOnRayDataframe(PandasDataframe):
    """
    The class implements the interface in ``PandasDataframe`` using Ray.

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
    _partition_mgr_cls = PandasOnRayDataframePartitionManager

    def _get_lengths(self, parts, axis):
        """
        Get list of  dimensions for all the provided parts.

        Parameters
        ----------
        parts : list
            List of parttions.
        axis : {0, 1}
            The axis along which to get the lengths (0 - length across rows or, 1 - width across columns).

        Returns
        -------
        list
        """
        if axis == Axis.ROW_WISE:
            dims = [part.length(False) for part in parts]
        else:
            dims = [part.width(False) for part in parts]
        return self._partition_mgr_cls.materialize_futures(dims)