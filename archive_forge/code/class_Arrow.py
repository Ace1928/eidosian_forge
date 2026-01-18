import itertools
from dataclasses import dataclass
from typing import Optional
import pyarrow as pa
import datasets
from datasets.table import table_cast
class Arrow(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = ArrowConfig

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
            if self.info.features is None:
                for file in itertools.chain.from_iterable(files):
                    with open(file, 'rb') as f:
                        self.info.features = datasets.Features.from_arrow_schema(pa.ipc.open_stream(f).schema)
                    break
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={'files': files}))
        return splits

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        if self.info.features is not None:
            pa_table = table_cast(pa_table, self.info.features.arrow_schema)
        return pa_table

    def _generate_tables(self, files):
        for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
            with open(file, 'rb') as f:
                try:
                    for batch_idx, record_batch in enumerate(pa.ipc.open_stream(f)):
                        pa_table = pa.Table.from_batches([record_batch])
                        yield (f'{file_idx}_{batch_idx}', self._cast_table(pa_table))
                except ValueError as e:
                    logger.error(f"Failed to read file '{file}' with error {type(e)}: {e}")
                    raise