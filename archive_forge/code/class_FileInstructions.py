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
@dataclass(frozen=True)
class FileInstructions:
    """The file instructions associated with a split ReadInstruction.

    Attributes:
        num_examples: `int`, The total number of examples
        file_instructions: List[dict(filename, skip, take)], the files information.
            The filenames contains the relative path, not absolute.
            skip/take indicates which example read in the file: `ds.slice(skip, take)`
    """
    num_examples: int
    file_instructions: List[dict]