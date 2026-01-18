import pickle
import io
import sys
import warnings
import contextlib
from .compressor import _ZFILE_PREFIX
from .compressor import _COMPRESSORS
def _get_prefixes_max_len():
    prefixes = [len(compressor.prefix) for compressor in _COMPRESSORS.values()]
    prefixes += [len(_ZFILE_PREFIX)]
    return max(prefixes)