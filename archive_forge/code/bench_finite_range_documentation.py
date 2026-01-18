import sys
import numpy as np
from numpy.testing import measure
from nibabel.volumeutils import finite_range  # NOQA
from .butils import print_git_title
Benchmarks for finite_range routine

Run benchmarks with::

    import nibabel as nib
    nib.bench()

Run this benchmark with::

    pytest -c <path>/benchmarks/pytest.benchmark.ini <path>/benchmarks/bench_finite_range.py
