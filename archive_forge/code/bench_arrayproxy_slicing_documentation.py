import gc
import itertools as it
from timeit import timeit
from unittest import mock
import numpy as np
import nibabel as nib
from nibabel.openers import HAVE_INDEXED_GZIP
from nibabel.tmpdirs import InTemporaryDirectory
from ..rstutils import rst_table
from .butils import print_git_title
Benchmarks for ArrayProxy slicing of gzipped and non-gzipped files

Run benchmarks with::

    import nibabel as nib
    nib.bench()

Run this benchmark with::

    pytest -c <path>/benchmarks/pytest.benchmark.ini <path>/benchmarks/bench_arrayproxy_slicing.py
