import os
from pyarrow.pandas_compat import _pandas_api  # noqa
from pyarrow.lib import (Codec, Table,  # noqa
import pyarrow.lib as ext
from pyarrow import _feather
from pyarrow._feather import FeatherError  # noqa: F401

        Read multiple Parquet files as a single pandas DataFrame

        Parameters
        ----------
        columns : List[str]
            Names of columns to read from the file
        use_threads : bool, default True
            Use multiple threads when converting to pandas

        Returns
        -------
        pandas.DataFrame
            Content of the file as a pandas DataFrame (of columns)
        