import os
from typing import BinaryIO, Optional, Union
import numpy as np
import pyarrow.parquet as pq
from .. import Audio, Dataset, Features, Image, NamedSplit, Value, config
from ..features.features import FeatureType, _visit
from ..formatting import query_table
from ..packaged_modules import _PACKAGED_DATASETS_MODULES
from ..packaged_modules.parquet.parquet import Parquet
from ..utils import tqdm as hf_tqdm
from ..utils.typing import NestedDataStructureLike, PathLike
from .abc import AbstractDatasetReader
class ParquetDatasetWriter:

    def __init__(self, dataset: Dataset, path_or_buf: Union[PathLike, BinaryIO], batch_size: Optional[int]=None, **parquet_writer_kwargs):
        self.dataset = dataset
        self.path_or_buf = path_or_buf
        self.batch_size = batch_size or get_writer_batch_size(dataset.features)
        self.parquet_writer_kwargs = parquet_writer_kwargs

    def write(self) -> int:
        batch_size = self.batch_size if self.batch_size else config.DEFAULT_MAX_BATCH_SIZE
        if isinstance(self.path_or_buf, (str, bytes, os.PathLike)):
            with open(self.path_or_buf, 'wb+') as buffer:
                written = self._write(file_obj=buffer, batch_size=batch_size, **self.parquet_writer_kwargs)
        else:
            written = self._write(file_obj=self.path_or_buf, batch_size=batch_size, **self.parquet_writer_kwargs)
        return written

    def _write(self, file_obj: BinaryIO, batch_size: int, **parquet_writer_kwargs) -> int:
        """Writes the pyarrow table as Parquet to a binary file handle.

        Caller is responsible for opening and closing the handle.
        """
        written = 0
        _ = parquet_writer_kwargs.pop('path_or_buf', None)
        schema = self.dataset.features.arrow_schema
        writer = pq.ParquetWriter(file_obj, schema=schema, **parquet_writer_kwargs)
        for offset in hf_tqdm(range(0, len(self.dataset), batch_size), unit='ba', desc='Creating parquet from Arrow format'):
            batch = query_table(table=self.dataset._data, key=slice(offset, offset + batch_size), indices=self.dataset._indices)
            writer.write_table(batch)
            written += batch.nbytes
        writer.close()
        return written