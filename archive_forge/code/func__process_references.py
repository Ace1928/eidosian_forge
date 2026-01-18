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
def _process_references(self, references, template_overrides=None):
    vers = references.get('version', None)
    if vers is None:
        self._process_references0(references)
    elif vers == 1:
        self._process_references1(references, template_overrides=template_overrides)
    else:
        raise ValueError(f'Unknown reference spec version: {vers}')