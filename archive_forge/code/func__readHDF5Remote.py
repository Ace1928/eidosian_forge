import copy
import os
import pickle
import warnings
import numpy as np
def _readHDF5Remote(self, fileName):
    proc = getattr(MetaArray, '_hdf5Process', None)
    if proc == False:
        raise Exception('remote read failed')
    if proc is None:
        from .. import multiprocess as mp
        proc = mp.Process(executable='/usr/bin/python')
        proc.setProxyOptions(deferGetattr=True)
        MetaArray._hdf5Process = proc
        MetaArray._h5py_metaarray = proc._import('pyqtgraph.metaarray')
    ma = MetaArray._h5py_metaarray.MetaArray(file=fileName)
    self._data = ma.asarray()._getValue()
    self._info = ma._info._getValue()