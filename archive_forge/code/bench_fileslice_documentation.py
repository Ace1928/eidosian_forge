import sys
from io import BytesIO
from timeit import timeit
import numpy as np
from ..fileslice import fileslice
from ..openers import ImageOpener
from ..optpkg import optional_package
from ..rstutils import rst_table
from ..tmpdirs import InTemporaryDirectory
Benchmarks for fileslicing

    import nibabel as nib
    nib.bench()

Run this benchmark with::

    pytest -c <path>/benchmarks/pytest.benchmark.ini <path>/benchmarks/bench_fileslice.py
