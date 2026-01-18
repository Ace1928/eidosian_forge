import os
from pyarrow.pandas_compat import _pandas_api  # noqa
from pyarrow.lib import (Codec, Table,  # noqa
import pyarrow.lib as ext
from pyarrow import _feather
from pyarrow._feather import FeatherError  # noqa: F401
class FeatherDataset:
    """
    Encapsulates details of reading a list of Feather files.

    Parameters
    ----------
    path_or_paths : List[str]
        A list of file names
    validate_schema : bool, default True
        Check that individual file schemas are all the same / compatible
    """

    def __init__(self, path_or_paths, validate_schema=True):
        self.paths = path_or_paths
        self.validate_schema = validate_schema

    def read_table(self, columns=None):
        """
        Read multiple feather files as a single pyarrow.Table

        Parameters
        ----------
        columns : List[str]
            Names of columns to read from the file

        Returns
        -------
        pyarrow.Table
            Content of the file as a table (of columns)
        """
        _fil = read_table(self.paths[0], columns=columns)
        self._tables = [_fil]
        self.schema = _fil.schema
        for path in self.paths[1:]:
            table = read_table(path, columns=columns)
            if self.validate_schema:
                self.validate_schemas(path, table)
            self._tables.append(table)
        return concat_tables(self._tables)

    def validate_schemas(self, piece, table):
        if not self.schema.equals(table.schema):
            raise ValueError('Schema in {!s} was different. \n{!s}\n\nvs\n\n{!s}'.format(piece, self.schema, table.schema))

    def read_pandas(self, columns=None, use_threads=True):
        """
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
        """
        return self.read_table(columns=columns).to_pandas(use_threads=use_threads)