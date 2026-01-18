import sys
from io import BytesIO
from timeit import timeit
import numpy as np
from ..fileslice import fileslice
from ..openers import ImageOpener
from ..optpkg import optional_package
from ..rstutils import rst_table
from ..tmpdirs import InTemporaryDirectory
def _slices_for_len(L):
    return (L // 2, slice(None, None, 1), slice(None, L // 2, 1), slice(None, None, L // 2))