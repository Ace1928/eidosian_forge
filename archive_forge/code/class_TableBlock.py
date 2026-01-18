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
class TableBlock(Table):
    """
    `TableBlock` is the allowed class inside a `ConcanetationTable`.
    Only `MemoryMappedTable` and `InMemoryTable` are `TableBlock`.
    This is because we don't want a `ConcanetationTable` made out of other `ConcanetationTables`.
    """
    pass