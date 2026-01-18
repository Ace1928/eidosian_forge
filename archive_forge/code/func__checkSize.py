import os
import zlib
import time  # noqa
import logging
import numpy as np
def _checkSize(self):
    arraylen = self.data.shape[0]
    if self._len >= arraylen:
        tmp = np.zeros((arraylen * 2,), dtype=np.uint8)
        tmp[:self._len] = self.data[:self._len]
        self.data = tmp