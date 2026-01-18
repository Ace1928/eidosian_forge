import base64
import collections
import io
import itertools
import logging
import math
import os
from functools import lru_cache
from typing import TYPE_CHECKING
import fsspec.core
from ..asyn import AsyncFileSystem
from ..callbacks import DEFAULT_CALLBACK
from ..core import filesystem, open, split_protocol
from ..utils import isfilelike, merge_offset_ranges, other_paths
def _get_chunk_sizes(self, field):
    """The number of chunks along each axis for a given field"""
    if field not in self.chunk_sizes:
        zarray = self.zmetadata[f'{field}/.zarray']
        size_ratio = [math.ceil(s / c) for s, c in zip(zarray['shape'], zarray['chunks'])]
        self.chunk_sizes[field] = size_ratio or [1]
    return self.chunk_sizes[field]