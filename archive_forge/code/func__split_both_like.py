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
def _split_both_like(result: List[List[TableBlock]], blocks: List[List[TableBlock]]) -> Tuple[List[List[TableBlock]], List[List[TableBlock]]]:
    """
            Make sure each row_block contain the same num_rows to be able to concatenate them on axis=1.

            To do so, we modify both blocks sets to have the same row_blocks boundaries.
            For example, if `result` has 2 row_blocks of 3 rows and `blocks` has 3 row_blocks of 2 rows,
            we modify both to have 4 row_blocks of size 2, 1, 1 and 2:

                    [ x   x   x | x   x   x ]
                +   [ y   y | y   y | y   y ]
                -----------------------------
                =   [ x   x | x | x | x   x ]
                    [ y   y | y | y | y   y ]

            """
    result, blocks = (list(result), list(blocks))
    new_result, new_blocks = ([], [])
    while result and blocks:
        if len(result[0][0]) > len(blocks[0][0]):
            new_blocks.append(blocks[0])
            sliced, result[0] = _slice_row_block(result[0], len(blocks.pop(0)[0]))
            new_result.append(sliced)
        elif len(result[0][0]) < len(blocks[0][0]):
            new_result.append(result[0])
            sliced, blocks[0] = _slice_row_block(blocks[0], len(result.pop(0)[0]))
            new_blocks.append(sliced)
        else:
            new_result.append(result.pop(0))
            new_blocks.append(blocks.pop(0))
    if result or blocks:
        raise ValueError("Failed to concatenate on axis=1 because tables don't have the same number of rows")
    return (new_result, new_blocks)