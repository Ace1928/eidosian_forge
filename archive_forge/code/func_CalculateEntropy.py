from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
def CalculateEntropy(self, input, alpha, num_threads=None):
    """Calculate sentence entropy"""
    if type(input) is list:
        if num_threads is None:
            num_threads = self._num_threads
        if num_threads is None or type(num_threads) is not int:
            raise RuntimeError('num_threads must be int')
        return self._CalculateEntropyBatch(input, alpha, num_threads)
    return self._CalculateEntropy(input, alpha)