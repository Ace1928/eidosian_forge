import sys
from io import BytesIO  # NOQA
import numpy as np
from numpy.testing import measure
from nibabel.volumeutils import array_to_file  # NOQA
from .butils import print_git_title
Benchmarks for array_to_file routine

Run benchmarks with::

    import nibabel as nib
    nib.bench()

Run this benchmark with::

    pytest -c <path>/benchmarks/pytest.benchmark.ini <path>/benchmarks/bench_array_to_file.py
