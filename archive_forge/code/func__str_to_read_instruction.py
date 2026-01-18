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
def _str_to_read_instruction(spec):
    """Returns ReadInstruction for given string."""
    res = _SUB_SPEC_RE.match(spec)
    if not res:
        raise ValueError(f'Unrecognized instruction format: {spec}')
    unit = '%' if res.group('from_pct') or res.group('to_pct') else 'abs'
    return ReadInstruction(split_name=res.group('split'), rounding=res.group('rounding'), from_=int(res.group('from')) if res.group('from') else None, to=int(res.group('to')) if res.group('to') else None, unit=unit)