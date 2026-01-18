from lib2to3.pgen2 import token
import os
import torch
import numpy as np
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
class _Writer(object):

    def __enter__(self):
        self._file = open(path, 'wb')
        self._file.write(cls._HDR_MAGIC)
        self._file.write(struct.pack('<Q', 1))
        self._file.write(struct.pack('<B', code(dtype)))
        return self

    @staticmethod
    def _get_pointers(sizes):
        dtype_size = dtype().itemsize
        address = 0
        pointers = []
        for size in sizes:
            pointers.append(address)
            address += size * dtype_size
        return pointers

    def write(self, sizes, doc_idx):
        pointers = self._get_pointers(sizes)
        self._file.write(struct.pack('<Q', len(sizes)))
        self._file.write(struct.pack('<Q', len(doc_idx)))
        sizes = np.array(sizes, dtype=np.int32)
        self._file.write(sizes.tobytes(order='C'))
        del sizes
        pointers = np.array(pointers, dtype=np.int64)
        self._file.write(pointers.tobytes(order='C'))
        del pointers
        doc_idx = np.array(doc_idx, dtype=np.int64)
        self._file.write(doc_idx.tobytes(order='C'))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()