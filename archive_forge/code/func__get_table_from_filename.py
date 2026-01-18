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
def _get_table_from_filename(self, filename_skip_take, **kwargs):
    """Returns a Dataset instance from given (filename, skip, take)."""
    filename, skip, take = (filename_skip_take['filename'], filename_skip_take['skip'] if 'skip' in filename_skip_take else None, filename_skip_take['take'] if 'take' in filename_skip_take else None)
    pa_table = pq.read_table(filename, memory_map=True)
    if skip is not None and take is not None and (not (skip == 0 and take == len(pa_table))):
        pa_table = pa_table.slice(skip, take)
    return pa_table