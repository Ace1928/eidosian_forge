from __future__ import annotations
import contextlib
import logging
import math
import shutil
import tempfile
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal
import numpy as np
import pandas as pd
import tlz as toolz
from pandas.api.types import is_numeric_dtype
from dask import config
from dask.base import compute, compute_as_if_collection, is_dask_collection, tokenize
from dask.dataframe import methods
from dask.dataframe._compat import PANDAS_GE_300
from dask.dataframe.core import (
from dask.dataframe.dispatch import (
from dask.dataframe.utils import UNKNOWN_CATEGORIES
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ShuffleLayer, SimpleShuffleLayer
from dask.sizeof import sizeof
from dask.utils import M, digit, get_default_shuffle_method
class maybe_buffered_partd:
    """
    If serialized, will return non-buffered partd. Otherwise returns a buffered partd
    """

    def __init__(self, encode_cls=None, buffer=True, tempdir=None):
        self.tempdir = tempdir or config.get('temporary_directory', None)
        self.buffer = buffer
        self.compression = config.get('dataframe.shuffle.compression', None)
        self.encode_cls = encode_cls
        if encode_cls is None:
            import partd
            self.encode_cls = partd.PandasBlocks

    def __reduce__(self):
        if self.tempdir:
            return (maybe_buffered_partd, (self.encode_cls, False, self.tempdir))
        else:
            return (maybe_buffered_partd, (self.encode_cls, False))

    def __call__(self, *args, **kwargs):
        import partd
        path = tempfile.mkdtemp(suffix='.partd', dir=self.tempdir)
        try:
            partd_compression = getattr(partd.compressed, self.compression) if self.compression else None
        except AttributeError as e:
            raise ImportError('Not able to import and load {} as compression algorithm.Please check if the library is installed and supported by Partd.'.format(self.compression)) from e
        file = partd.File(path)
        partd.file.cleanup_files.append(path)
        if partd_compression:
            file = partd_compression(file)
        if self.buffer:
            return self.encode_cls(partd.Buffer(partd.Dict(), file))
        else:
            return self.encode_cls(file)