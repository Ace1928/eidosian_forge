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
def _memory_mapped_arrow_table_from_file(filename: str) -> pa.Table:
    opened_stream = _memory_mapped_record_batch_reader_from_file(filename)
    pa_table = opened_stream.read_all()
    return pa_table