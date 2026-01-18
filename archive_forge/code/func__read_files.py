import copy
import math
import os
import re
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.contrib.concurrent import thread_map
from .download.download_config import DownloadConfig
from .naming import _split_re, filenames_for_dataset_split
from .table import InMemoryTable, MemoryMappedTable, Table, concat_tables
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.file_utils import cached_path
def _read_files(self, files, in_memory=False) -> Table:
    """Returns Dataset for given file instructions.

        Args:
            files: List[dict(filename, skip, take)], the files information.
                The filenames contain the absolute path, not relative.
                skip/take indicates which example read in the file: `ds.slice(skip, take)`
            in_memory (bool, default False): Whether to copy the data in-memory.
        """
    if len(files) == 0 or not all((isinstance(f, dict) for f in files)):
        raise ValueError('please provide valid file informations')
    files = copy.deepcopy(files)
    for f in files:
        f['filename'] = os.path.join(self._path, f['filename'])
    pa_tables = thread_map(partial(self._get_table_from_filename, in_memory=in_memory), files, tqdm_class=hf_tqdm, desc='Loading dataset shards', disable=len(files) <= 16 or None)
    pa_tables = [t for t in pa_tables if len(t) > 0]
    if not pa_tables and (self._info is None or self._info.features is None):
        raise ValueError('Tried to read an empty table. Please specify at least info.features to create an empty table with the right type.')
    pa_tables = pa_tables or [InMemoryTable.from_batches([], schema=pa.schema(self._info.features.type))]
    pa_table = concat_tables(pa_tables) if len(pa_tables) != 1 else pa_tables[0]
    return pa_table