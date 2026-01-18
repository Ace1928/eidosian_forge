import sys
from io import BytesIO
import numpy as np
from numpy.testing import measure
from .. import Nifti1Image
from .butils import print_git_title
Benchmarks for load and save of image arrays

Run benchmarks with::

    import nibabel as nib
    nib.bench()

Run this benchmark with::

    pytest -c <path>/benchmarks/pytest.benchmark.ini <path>/benchmarks/bench_load_save.py
