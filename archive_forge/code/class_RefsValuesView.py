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
class RefsValuesView(collections.abc.ValuesView):

    def __iter__(self):
        for val in self._mapping.zmetadata.values():
            yield json.dumps(val).encode()
        yield from self._mapping._items.values()
        for field in self._mapping.listdir():
            chunk_sizes = self._mapping._get_chunk_sizes(field)
            if len(chunk_sizes) == 0:
                yield self._mapping[field + '/0']
                continue
            yield from self._mapping._generate_all_records(field)