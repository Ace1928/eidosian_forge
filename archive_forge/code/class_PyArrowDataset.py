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
class PyArrowDataset(ColumnStoreDataset):

    def _init_dataset(self):
        from pyarrow.parquet import ParquetDataset
        return ParquetDataset(self.fs_path, filesystem=self.fs, use_legacy_dataset=False)

    @property
    def pandas_metadata(self):
        return self.dataset.schema.pandas_metadata

    @property
    def columns(self):
        return self.dataset.schema.names

    @property
    def engine(self):
        return 'pyarrow'

    @property
    def row_groups_per_file(self):
        from pyarrow.parquet import ParquetFile
        if self._row_groups_per_file is None:
            row_groups_per_file = []
            for file in self.files:
                with self.fs.open(file) as f:
                    row_groups = ParquetFile(f).num_row_groups
                    row_groups_per_file.append(row_groups)
            self._row_groups_per_file = row_groups_per_file
        return self._row_groups_per_file

    @property
    def files(self):
        if self._files is None:
            try:
                files = self.dataset.files
            except AttributeError:
                files = self.dataset._dataset.files
            self._files = self._get_files(files)
        return self._files

    def to_pandas_dataframe(self, columns):
        from pyarrow.parquet import read_table
        return read_table(self._fs_path, columns=columns, filesystem=self.fs).to_pandas()