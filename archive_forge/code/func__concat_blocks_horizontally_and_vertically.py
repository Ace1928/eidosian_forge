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
@classmethod
def _concat_blocks_horizontally_and_vertically(cls, blocks: List[List[TableBlock]]) -> pa.Table:
    pa_tables_to_concat_vertically = []
    for i, tables in enumerate(blocks):
        if not tables:
            continue
        pa_table_horizontally_concatenated = cls._concat_blocks(tables, axis=1)
        pa_tables_to_concat_vertically.append(pa_table_horizontally_concatenated)
    return cls._concat_blocks(pa_tables_to_concat_vertically, axis=0)