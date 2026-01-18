from __future__ import annotations
import logging
import os
import time
import traceback
from collections.abc import Iterable
from glob import glob
from typing import TYPE_CHECKING, Any, ClassVar
import numpy as np
from xarray.conventions import cf_encoder
from xarray.core import indexing
from xarray.core.utils import FrozenDict, NdimSizeLenMixin, is_remote_uri
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
class ArrayWriter:
    __slots__ = ('sources', 'targets', 'regions', 'lock')

    def __init__(self, lock=None):
        self.sources = []
        self.targets = []
        self.regions = []
        self.lock = lock

    def add(self, source, target, region=None):
        if is_chunked_array(source):
            self.sources.append(source)
            self.targets.append(target)
            self.regions.append(region)
        elif region:
            target[region] = source
        else:
            target[...] = source

    def sync(self, compute=True, chunkmanager_store_kwargs=None):
        if self.sources:
            chunkmanager = get_chunked_array_type(*self.sources)
            if chunkmanager_store_kwargs is None:
                chunkmanager_store_kwargs = {}
            delayed_store = chunkmanager.store(self.sources, self.targets, lock=self.lock, compute=compute, flush=True, regions=self.regions, **chunkmanager_store_kwargs)
            self.sources = []
            self.targets = []
            self.regions = []
            return delayed_store