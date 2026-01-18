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
def _prot_in_references(path, references):
    ref = references.get(path)
    if isinstance(ref, (list, tuple)):
        return split_protocol(ref[0])[0] if ref[0] else ref[0]