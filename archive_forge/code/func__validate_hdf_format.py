import pandas
from modin.core.io.column_stores.column_store_dispatcher import ColumnStoreDispatcher
@classmethod
def _validate_hdf_format(cls, path_or_buf):
    """
        Validate `path_or_buf` and then return `table_type` parameter of store group attribute.

        Parameters
        ----------
        path_or_buf : str, buffer or path object
            Path to the file to open, or an open :class:`pandas.HDFStore` object.

        Returns
        -------
        str
            `table_type` parameter of store group attribute.
        """
    s = pandas.HDFStore(path_or_buf)
    groups = s.groups()
    if len(groups) == 0:
        raise ValueError('No dataset in HDF5 file.')
    candidate_only_group = groups[0]
    format = getattr(candidate_only_group._v_attrs, 'table_type', None)
    s.close()
    return format