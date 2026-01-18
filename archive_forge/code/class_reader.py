import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
class reader:

    def __init__(self):
        self.use_threads = use_threads

    def _patch_kwargs(self, kwargs):
        if 'use_threads' in kwargs:
            raise Exception('Invalid use of dataset_reader, do not specify use_threads')
        kwargs['use_threads'] = use_threads

    def to_table(self, dataset, **kwargs):
        self._patch_kwargs(kwargs)
        return dataset.to_table(**kwargs)

    def to_batches(self, dataset, **kwargs):
        self._patch_kwargs(kwargs)
        return dataset.to_batches(**kwargs)

    def scanner(self, dataset, **kwargs):
        self._patch_kwargs(kwargs)
        return dataset.scanner(**kwargs)

    def head(self, dataset, num_rows, **kwargs):
        self._patch_kwargs(kwargs)
        return dataset.head(num_rows, **kwargs)

    def take(self, dataset, indices, **kwargs):
        self._patch_kwargs(kwargs)
        return dataset.take(indices, **kwargs)

    def count_rows(self, dataset, **kwargs):
        self._patch_kwargs(kwargs)
        return dataset.count_rows(**kwargs)