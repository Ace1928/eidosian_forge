from __future__ import annotations
import enum
import functools
import operator
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import timedelta
from html import escape
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
from xarray.core import duck_array_ops
from xarray.core.nputils import NumpyVIndexAdapter
from xarray.core.options import OPTIONS
from xarray.core.types import T_Xarray
from xarray.core.utils import (
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import array_type, integer_types, is_chunked_array
def _chunked_array_with_chunks_hint(array, chunks, chunkmanager: ChunkManagerEntrypoint[Any]):
    """Create a chunked array using the chunks hint for dimensions of size > 1."""
    if len(chunks) < array.ndim:
        raise ValueError('not enough chunks in hint')
    new_chunks = []
    for chunk, size in zip(chunks, array.shape):
        new_chunks.append(chunk if size > 1 else (1,))
    return chunkmanager.from_array(array, new_chunks)