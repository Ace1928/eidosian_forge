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
def _combine_list_array_offsets_with_mask(array: pa.ListArray) -> pa.Array:
    """Add the null bitmap to the offsets of a `pa.ListArray`."""
    offsets = array.offsets
    if array.null_count > 0:
        offsets = pa.concat_arrays([pc.replace_with_mask(offsets[:-1], array.is_null(), pa.nulls(len(array), pa.int32())), offsets[-1:]])
    return offsets