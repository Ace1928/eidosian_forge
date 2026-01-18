import io
import itertools
import json
from dataclasses import dataclass
from typing import Optional
import pyarrow as pa
import pyarrow.json as paj
import datasets
from datasets.table import table_cast
from datasets.utils.file_utils import readline
class Json(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = JsonConfig

    def _info(self):
        if self.config.block_size is not None:
            logger.warning('The JSON loader parameter `block_size` is deprecated. Please use `chunksize` instead')
            self.config.chunksize = self.config.block_size
        if self.config.use_threads is not True:
            logger.warning("The JSON loader parameter `use_threads` is deprecated and doesn't have any effect anymore.")
        if self.config.newlines_in_values is not None:
            raise ValueError('The JSON loader parameter `newlines_in_values` is no longer supported')
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
            for column_name in set(self.config.features) - set(pa_table.column_names):
                type = self.config.features.arrow_schema.field(column_name).type
                pa_table = pa_table.append_column(column_name, pa.array([None] * len(pa_table), type=type))
            pa_table = table_cast(pa_table, self.config.features.arrow_schema)
        return pa_table

    def _generate_tables(self, files):
        for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
            if self.config.field is not None:
                with open(file, encoding=self.config.encoding, errors=self.config.encoding_errors) as f:
                    dataset = json.load(f)
                dataset = dataset[self.config.field]
                if isinstance(dataset, (list, tuple)):
                    keys = set().union(*[row.keys() for row in dataset])
                    mapping = {col: [row.get(col) for row in dataset] for col in keys}
                else:
                    mapping = dataset
                pa_table = pa.Table.from_pydict(mapping)
                yield (file_idx, self._cast_table(pa_table))
            else:
                with open(file, 'rb') as f:
                    batch_idx = 0
                    block_size = max(self.config.chunksize // 32, 16 << 10)
                    encoding_errors = self.config.encoding_errors if self.config.encoding_errors is not None else 'strict'
                    while True:
                        batch = f.read(self.config.chunksize)
                        if not batch:
                            break
                        try:
                            batch += f.readline()
                        except (AttributeError, io.UnsupportedOperation):
                            batch += readline(f)
                        if self.config.encoding != 'utf-8':
                            batch = batch.decode(self.config.encoding, errors=encoding_errors).encode('utf-8')
                        try:
                            while True:
                                try:
                                    pa_table = paj.read_json(io.BytesIO(batch), read_options=paj.ReadOptions(block_size=block_size))
                                    break
                                except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
                                    if isinstance(e, pa.ArrowInvalid) and 'straddling' not in str(e) or block_size > len(batch):
                                        raise
                                    else:
                                        logger.debug(f"Batch of {len(batch)} bytes couldn't be parsed with block_size={block_size}. Retrying with block_size={block_size * 2}.")
                                        block_size *= 2
                        except pa.ArrowInvalid as e:
                            try:
                                with open(file, encoding=self.config.encoding, errors=self.config.encoding_errors) as f:
                                    dataset = json.load(f)
                            except json.JSONDecodeError:
                                logger.error(f"Failed to read file '{file}' with error {type(e)}: {e}")
                                raise e
                            if isinstance(dataset, list):
                                try:
                                    if dataset and isinstance(dataset[0], str):
                                        pa_table_names = list(self.config.features) if self.config.features is not None else ['text']
                                        pa_table = pa.Table.from_arrays([pa.array(dataset)], names=pa_table_names)
                                    else:
                                        keys = set().union(*[row.keys() for row in dataset])
                                        mapping = {col: [row.get(col) for row in dataset] for col in keys}
                                        pa_table = pa.Table.from_pydict(mapping)
                                except (pa.ArrowInvalid, AttributeError) as e:
                                    logger.error(f"Failed to read file '{file}' with error {type(e)}: {e}")
                                    raise ValueError(f'Not able to read records in the JSON file at {file}.') from None
                                yield (file_idx, self._cast_table(pa_table))
                                break
                            else:
                                logger.error(f"Failed to read file '{file}' with error {type(e)}: {e}")
                                raise ValueError(f"Not able to read records in the JSON file at {file}. You should probably indicate the field of the JSON file containing your records. This JSON file contain the following fields: {str(list(dataset.keys()))}. Select the correct one and provide it as `field='XXX'` to the dataset loading method. ") from None
                        yield ((file_idx, batch_idx), self._cast_table(pa_table))
                        batch_idx += 1