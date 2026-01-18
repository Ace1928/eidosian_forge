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
def _load_one_key(self, key):
    """Get the reference for one key

        Returns bytes, one-element list or three-element list.
        """
    if key in self._items:
        return self._items[key]
    elif key in self.zmetadata:
        return json.dumps(self.zmetadata[key]).encode()
    elif '/' not in key or self._is_meta(key):
        raise KeyError(key)
    field, _ = key.rsplit('/', 1)
    record, ri, chunk_size = self._key_to_record(key)
    maybe = self._items.get((field, record), {}).get(ri, False)
    if maybe is None:
        raise KeyError
    elif maybe:
        return maybe
    elif chunk_size == 0:
        return b''
    try:
        refs = self.open_refs(field, record)
    except (ValueError, TypeError, FileNotFoundError):
        raise KeyError(key)
    columns = ['path', 'offset', 'size', 'raw']
    selection = [refs[c][ri] if c in refs else None for c in columns]
    raw = selection[-1]
    if raw is not None:
        return raw
    if selection[0] is None:
        raise KeyError('This reference does not exist or has been deleted')
    if selection[1:3] == [0, 0]:
        return selection[:1]
    return selection[:3]