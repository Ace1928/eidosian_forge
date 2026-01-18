from pandas.io.common import stringify_path
from modin.core.io.column_stores.column_store_dispatcher import ColumnStoreDispatcher
from modin.core.io.file_dispatcher import OpenFile
from modin.utils import import_optional_dependency

        Read data from the file path, returning a query compiler.

        Parameters
        ----------
        path : str or file-like object
            The filepath of the feather file.
        columns : array-like, optional
            Columns to read from file. If not provided, all columns are read.
        **kwargs : dict
            `read_feather` function kwargs.

        Returns
        -------
        BaseQueryCompiler
            Query compiler with imported data for further processing.

        Notes
        -----
        `PyArrow` engine and local files only are supported for now,
        multi threading is set to False by default.
        PyArrow feather is used. Please refer to the documentation here
        https://arrow.apache.org/docs/python/api.html#feather-format
        