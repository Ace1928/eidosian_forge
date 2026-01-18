import copy
import os
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from . import config
from .utils.logging import get_logger
def _memory_mapped_record_batch_reader_from_file(filename: str) -> pa.RecordBatchStreamReader:
    memory_mapped_stream = pa.memory_map(filename)
    return pa.ipc.open_stream(memory_mapped_stream)