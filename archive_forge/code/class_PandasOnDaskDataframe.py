from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
from ..partitioning.partition_manager import PandasOnDaskDataframePartitionManager
class PandasOnDaskDataframe(PandasDataframe):
    """
    The class implements the interface in ``PandasDataframe``.

    Parameters
    ----------
    partitions : np.ndarray
        A 2D NumPy array of partitions.
    index : sequence
        The index for the dataframe. Converted to a pandas.Index.
    columns : sequence
        The columns object for the dataframe. Converted to a pandas.Index.
    row_lengths : list, optional
        The length of each partition in the rows. The "height" of
        each of the block partitions. Is computed if not provided.
    column_widths : list, optional
        The width of each partition in the columns. The "width" of
        each of the block partitions. Is computed if not provided.
    dtypes : pandas.Series, optional
        The data types for the dataframe columns.
    """
    _partition_mgr_cls = PandasOnDaskDataframePartitionManager

    @classmethod
    def reconnect(cls, address, attributes):
        try:
            from distributed import default_client
            default_client()
        except ValueError:
            from distributed import Client
            _ = Client(address)
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj

    def __reduce__(self):
        from distributed import default_client
        address = default_client().scheduler_info()['address']
        return (self.reconnect, (address, self.__dict__))