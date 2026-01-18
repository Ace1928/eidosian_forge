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
def _consolidate_blocks(cls, blocks: TableBlockContainer) -> TableBlockContainer:
    if isinstance(blocks, TableBlock):
        return blocks
    elif isinstance(blocks[0], TableBlock):
        return cls._merge_blocks(blocks, axis=0)
    else:
        return cls._merge_blocks(blocks)