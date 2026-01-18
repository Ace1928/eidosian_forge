import errno
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from . import config
from .features import Features, Image, Value
from .features.features import (
from .filesystems import is_remote_filesystem
from .info import DatasetInfo
from .keyhash import DuplicatedKeysError, KeyHasher
from .table import array_cast, cast_array_to_feature, embed_table_storage, table_cast
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.file_utils import hash_url_to_filename
from .utils.py_utils import asdict, first_non_null_value
class ArrowWriter:
    """Shuffles and writes Examples to Arrow files."""
    _WRITER_CLASS = pa.RecordBatchStreamWriter

    def __init__(self, schema: Optional[pa.Schema]=None, features: Optional[Features]=None, path: Optional[str]=None, stream: Optional[pa.NativeFile]=None, fingerprint: Optional[str]=None, writer_batch_size: Optional[int]=None, hash_salt: Optional[str]=None, check_duplicates: Optional[bool]=False, disable_nullable: bool=False, update_features: bool=False, with_metadata: bool=True, unit: str='examples', embed_local_files: bool=False, storage_options: Optional[dict]=None):
        if path is None and stream is None:
            raise ValueError('At least one of path and stream must be provided.')
        if features is not None:
            self._features = features
            self._schema = None
        elif schema is not None:
            self._schema: pa.Schema = schema
            self._features = Features.from_arrow_schema(self._schema)
        else:
            self._features = None
            self._schema = None
        if hash_salt is not None:
            self._hasher = KeyHasher(hash_salt)
        else:
            self._hasher = KeyHasher('')
        self._check_duplicates = check_duplicates
        self._disable_nullable = disable_nullable
        if stream is None:
            fs_token_paths = fsspec.get_fs_token_paths(path, storage_options=storage_options)
            self._fs: fsspec.AbstractFileSystem = fs_token_paths[0]
            self._path = fs_token_paths[2][0] if not is_remote_filesystem(self._fs) else self._fs.unstrip_protocol(fs_token_paths[2][0])
            self.stream = self._fs.open(fs_token_paths[2][0], 'wb')
            self._closable_stream = True
        else:
            self._fs = None
            self._path = None
            self.stream = stream
            self._closable_stream = False
        self.fingerprint = fingerprint
        self.disable_nullable = disable_nullable
        self.writer_batch_size = writer_batch_size or config.DEFAULT_MAX_BATCH_SIZE
        self.update_features = update_features
        self.with_metadata = with_metadata
        self.unit = unit
        self.embed_local_files = embed_local_files
        self._num_examples = 0
        self._num_bytes = 0
        self.current_examples: List[Tuple[Dict[str, Any], str]] = []
        self.current_rows: List[pa.Table] = []
        self.pa_writer: Optional[pa.RecordBatchStreamWriter] = None
        self.hkey_record = []

    def __len__(self):
        """Return the number of writed and staged examples"""
        return self._num_examples + len(self.current_examples) + len(self.current_rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.pa_writer:
            try:
                self.pa_writer.close()
            except Exception:
                pass
        if self._closable_stream and (not self.stream.closed):
            self.stream.close()

    def _build_writer(self, inferred_schema: pa.Schema):
        schema = self.schema
        inferred_features = Features.from_arrow_schema(inferred_schema)
        if self._features is not None:
            if self.update_features:
                fields = {field.name: field for field in self._features.type}
                for inferred_field in inferred_features.type:
                    name = inferred_field.name
                    if name in fields:
                        if inferred_field == fields[name]:
                            inferred_features[name] = self._features[name]
                self._features = inferred_features
                schema: pa.Schema = inferred_schema
        else:
            self._features = inferred_features
            schema: pa.Schema = inferred_features.arrow_schema
        if self.disable_nullable:
            schema = pa.schema((pa.field(field.name, field.type, nullable=False) for field in schema))
        if self.with_metadata:
            schema = schema.with_metadata(self._build_metadata(DatasetInfo(features=self._features), self.fingerprint))
        else:
            schema = schema.with_metadata({})
        self._schema = schema
        self.pa_writer = self._WRITER_CLASS(self.stream, schema)

    @property
    def schema(self):
        _schema = self._schema if self._schema is not None else pa.schema(self._features.type) if self._features is not None else None
        if self._disable_nullable and _schema is not None:
            _schema = pa.schema((pa.field(field.name, field.type, nullable=False) for field in _schema))
        return _schema if _schema is not None else []

    @staticmethod
    def _build_metadata(info: DatasetInfo, fingerprint: Optional[str]=None) -> Dict[str, str]:
        info_keys = ['features']
        info_as_dict = asdict(info)
        metadata = {}
        metadata['info'] = {key: info_as_dict[key] for key in info_keys}
        if fingerprint is not None:
            metadata['fingerprint'] = fingerprint
        return {'huggingface': json.dumps(metadata)}

    def write_examples_on_file(self):
        """Write stored examples from the write-pool of examples. It makes a table out of the examples and write it."""
        if not self.current_examples:
            return
        if self.schema:
            schema_cols = set(self.schema.names)
            examples_cols = self.current_examples[0][0].keys()
            common_cols = [col for col in self.schema.names if col in examples_cols]
            extra_cols = [col for col in examples_cols if col not in schema_cols]
            cols = common_cols + extra_cols
        else:
            cols = list(self.current_examples[0][0])
        batch_examples = {}
        for col in cols:
            if all((isinstance(row[0][col], (pa.Array, pa.ChunkedArray)) for row in self.current_examples)):
                arrays = [row[0][col] for row in self.current_examples]
                arrays = [chunk for array in arrays for chunk in (array.chunks if isinstance(array, pa.ChunkedArray) else [array])]
                batch_examples[col] = pa.concat_arrays(arrays)
            else:
                batch_examples[col] = [row[0][col].to_pylist()[0] if isinstance(row[0][col], (pa.Array, pa.ChunkedArray)) else row[0][col] for row in self.current_examples]
        self.write_batch(batch_examples=batch_examples)
        self.current_examples = []

    def write_rows_on_file(self):
        """Write stored rows from the write-pool of rows. It concatenates the single-row tables and it writes the resulting table."""
        if not self.current_rows:
            return
        table = pa.concat_tables(self.current_rows)
        self.write_table(table)
        self.current_rows = []

    def write(self, example: Dict[str, Any], key: Optional[Union[str, int, bytes]]=None, writer_batch_size: Optional[int]=None):
        """Add a given (Example,Key) pair to the write-pool of examples which is written to file.

        Args:
            example: the Example to add.
            key: Optional, a unique identifier(str, int or bytes) associated with each example
        """
        if self._check_duplicates:
            hash = self._hasher.hash(key)
            self.current_examples.append((example, hash))
            self.hkey_record.append((hash, key))
        else:
            self.current_examples.append((example, ''))
        if writer_batch_size is None:
            writer_batch_size = self.writer_batch_size
        if writer_batch_size is not None and len(self.current_examples) >= writer_batch_size:
            if self._check_duplicates:
                self.check_duplicate_keys()
                self.hkey_record = []
            self.write_examples_on_file()

    def check_duplicate_keys(self):
        """Raises error if duplicates found in a batch"""
        tmp_record = set()
        for hash, key in self.hkey_record:
            if hash in tmp_record:
                duplicate_key_indices = [str(self._num_examples + index) for index, (duplicate_hash, _) in enumerate(self.hkey_record) if duplicate_hash == hash]
                raise DuplicatedKeysError(key, duplicate_key_indices)
            else:
                tmp_record.add(hash)

    def write_row(self, row: pa.Table, writer_batch_size: Optional[int]=None):
        """Add a given single-row Table to the write-pool of rows which is written to file.

        Args:
            row: the row to add.
        """
        if len(row) != 1:
            raise ValueError(f'Only single-row pyarrow tables are allowed but got table with {len(row)} rows.')
        self.current_rows.append(row)
        if writer_batch_size is None:
            writer_batch_size = self.writer_batch_size
        if writer_batch_size is not None and len(self.current_rows) >= writer_batch_size:
            self.write_rows_on_file()

    def write_batch(self, batch_examples: Dict[str, List], writer_batch_size: Optional[int]=None):
        """Write a batch of Example to file.
        Ignores the batch if it appears to be empty,
        preventing a potential schema update of unknown types.

        Args:
            batch_examples: the batch of examples to add.
        """
        if batch_examples and len(next(iter(batch_examples.values()))) == 0:
            return
        features = None if self.pa_writer is None and self.update_features else self._features
        try_features = self._features if self.pa_writer is None and self.update_features else None
        arrays = []
        inferred_features = Features()
        if self.schema:
            schema_cols = set(self.schema.names)
            batch_cols = batch_examples.keys()
            common_cols = [col for col in self.schema.names if col in batch_cols]
            extra_cols = [col for col in batch_cols if col not in schema_cols]
            cols = common_cols + extra_cols
        else:
            cols = list(batch_examples)
        for col in cols:
            col_values = batch_examples[col]
            col_type = features[col] if features else None
            if isinstance(col_values, (pa.Array, pa.ChunkedArray)):
                array = cast_array_to_feature(col_values, col_type) if col_type is not None else col_values
                arrays.append(array)
                inferred_features[col] = generate_from_arrow_type(col_values.type)
            else:
                col_try_type = try_features[col] if try_features is not None and col in try_features else None
                typed_sequence = OptimizedTypedSequence(col_values, type=col_type, try_type=col_try_type, col=col)
                arrays.append(pa.array(typed_sequence))
                inferred_features[col] = typed_sequence.get_inferred_type()
        schema = inferred_features.arrow_schema if self.pa_writer is None else self.schema
        pa_table = pa.Table.from_arrays(arrays, schema=schema)
        self.write_table(pa_table, writer_batch_size)

    def write_table(self, pa_table: pa.Table, writer_batch_size: Optional[int]=None):
        """Write a Table to file.

        Args:
            example: the Table to add.
        """
        if writer_batch_size is None:
            writer_batch_size = self.writer_batch_size
        if self.pa_writer is None:
            self._build_writer(inferred_schema=pa_table.schema)
        pa_table = pa_table.combine_chunks()
        pa_table = table_cast(pa_table, self._schema)
        if self.embed_local_files:
            pa_table = embed_table_storage(pa_table)
        self._num_bytes += pa_table.nbytes
        self._num_examples += pa_table.num_rows
        self.pa_writer.write_table(pa_table, writer_batch_size)

    def finalize(self, close_stream=True):
        self.write_rows_on_file()
        if self._check_duplicates:
            self.check_duplicate_keys()
            self.hkey_record = []
        self.write_examples_on_file()
        if self.pa_writer is None and self.schema:
            self._build_writer(self.schema)
        if self.pa_writer is not None:
            self.pa_writer.close()
            self.pa_writer = None
            if close_stream:
                self.stream.close()
        else:
            if close_stream:
                self.stream.close()
            raise SchemaInferenceError('Please pass `features` or at least one example when writing data')
        logger.debug(f'Done writing {self._num_examples} {self.unit} in {self._num_bytes} bytes {(self._path if self._path else '')}.')
        return (self._num_examples, self._num_bytes)