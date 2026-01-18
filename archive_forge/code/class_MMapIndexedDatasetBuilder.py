import os
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
import numpy as np
import torch
class MMapIndexedDatasetBuilder(object):

    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, 'wb')
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    @property
    def dtype(self):
        return self._dtype

    def add_item(self, np_array):
        assert isinstance(np_array, np.ndarray) and np_array.dtype == self.dtype
        self._data_file.write(np_array.tobytes(order='C'))
        self._sizes.append(np_array.size)

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def merge_file_(self, another_file):
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype
        for size in index.sizes:
            self._sizes.append(size)
        with open(data_file_path(another_file), 'rb') as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        self._data_file.close()
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)