import json
import os
import re
from typing import TYPE_CHECKING
import fsspec
import numpy as np
import pandas
import pandas._libs.lib as lib
from fsspec.core import url_to_fs
from fsspec.spec import AbstractBufferedFile
from packaging import version
from pandas.io.common import stringify_path
from modin.config import MinPartitionSize, NPartitions
from modin.core.io.column_stores.column_store_dispatcher import ColumnStoreDispatcher
from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings
@_inherit_docstrings(ColumnStoreDataset)
class FastParquetDataset(ColumnStoreDataset):

    def _init_dataset(self):
        from fastparquet import ParquetFile
        return ParquetFile(self.fs_path, fs=self.fs)

    @property
    def pandas_metadata(self):
        if 'pandas' not in self.dataset.key_value_metadata:
            return {}
        return json.loads(self.dataset.key_value_metadata['pandas'])

    @property
    def columns(self):
        return self.dataset.columns

    @property
    def engine(self):
        return 'fastparquet'

    @property
    def row_groups_per_file(self):
        from fastparquet import ParquetFile
        if self._row_groups_per_file is None:
            row_groups_per_file = []
            for file in self.files:
                with self.fs.open(file) as f:
                    row_groups = ParquetFile(f).info['row_groups']
                    row_groups_per_file.append(row_groups)
            self._row_groups_per_file = row_groups_per_file
        return self._row_groups_per_file

    @property
    def files(self):
        if self._files is None:
            self._files = self._get_files(self._get_fastparquet_files())
        return self._files

    def to_pandas_dataframe(self, columns):
        return self.dataset.to_pandas(columns=columns)

    def _get_fastparquet_files(self):
        if '*' in self.path:
            files = self.fs.glob(self.path)
        elif self.fs.isfile(self.path):
            files = self.fs.find(self.path)
        else:
            files = [f for f in self.fs.find(self.path) if f.endswith('.parquet') or f.endswith('.parq')]
        return files