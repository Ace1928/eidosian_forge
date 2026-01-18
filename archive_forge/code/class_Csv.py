import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
import pandas as pd
import pyarrow as pa
import datasets
import datasets.config
from datasets.features.features import require_storage_cast
from datasets.table import table_cast
from datasets.utils.py_utils import Literal
class Csv(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = CsvConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(f'At least one data file must be specified, but got data_files={self.config.data_files}')
        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            if isinstance(files, str):
                files = [files]
            files = [dl_manager.iter_files(file) for file in files]
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'files': files})]
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            files = [dl_manager.iter_files(file) for file in files]
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={'files': files}))
        return splits

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        if self.config.features is not None:
            schema = self.config.features.arrow_schema
            if all((not require_storage_cast(feature) for feature in self.config.features.values())):
                pa_table = pa.Table.from_arrays([pa_table[field.name] for field in schema], schema=schema)
            else:
                pa_table = table_cast(pa_table, schema)
        return pa_table

    def _generate_tables(self, files):
        schema = self.config.features.arrow_schema if self.config.features else None
        dtype = {name: dtype.to_pandas_dtype() if not require_storage_cast(feature) else object for name, dtype, feature in zip(schema.names, schema.types, self.config.features.values())} if schema is not None else None
        for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
            csv_file_reader = pd.read_csv(file, iterator=True, dtype=dtype, **self.config.pd_read_csv_kwargs)
            try:
                for batch_idx, df in enumerate(csv_file_reader):
                    pa_table = pa.Table.from_pandas(df)
                    yield ((file_idx, batch_idx), self._cast_table(pa_table))
            except ValueError as e:
                logger.error(f"Failed to read file '{file}' with error {type(e)}: {e}")
                raise