import hashlib
import logging
import numpy as np
import os
import pandas as pd
import tarfile
import tempfile
import six
import shutil
from .core import PATH_TYPES, fspath
def _load_numeric_only_dataset(path, row_count, column_count, sep='\t'):
    dataset = np.zeros((row_count, column_count), dtype=np.float32, order='F')
    with open(path, 'rb') as f:
        for line_idx, line in enumerate(f):
            row = np.fromstring(line, dtype=np.float32, sep=sep)
            assert row.size == column_count, 'got too many columns at line %d (expected %d columns, got %d)' % (line_idx + 1, column_count, row.size)
            dataset[line_idx][:] = row
    assert line_idx + 1 == row_count, 'got too many lines (expected %d lines, got %d)' % (row_count, line_idx + 1)
    return pd.DataFrame(dataset)