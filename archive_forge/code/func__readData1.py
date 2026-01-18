import copy
import os
import pickle
import warnings
import numpy as np
def _readData1(self, fd, meta, mmap=False, **kwds):
    frameSize = 1
    for ax in meta['info']:
        if 'values_len' in ax:
            ax['values'] = np.frombuffer(fd.read(ax['values_len']), dtype=ax['values_type'])
            frameSize *= ax['values_len']
            del ax['values_len']
            del ax['values_type']
    self._info = meta['info']
    if not kwds.get('readAllData', True):
        return
    if mmap:
        subarr = np.memmap(fd, dtype=meta['type'], mode='r', shape=meta['shape'])
    else:
        subarr = np.frombuffer(fd.read(), dtype=meta['type'])
        subarr.shape = meta['shape']
    self._data = subarr